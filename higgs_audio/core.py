"""
Core classes for Higgs Audio voice cloning and TTS generation.
"""

import os
import numpy as np
import torch
from typing import Optional, Union, Dict, Any, List
from pathlib import Path

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample, ChatMLDatasetSample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from dataclasses import asdict
import tqdm
import langid
import jieba


class VoiceProfile:
    """Represents a voice profile extracted from reference audio."""
    
    def __init__(self, tokens: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self.tokens = tokens
        self.metadata = metadata or {}
    
    @property
    def shape(self) -> tuple:
        return self.tokens.shape
    
    @property
    def size(self) -> int:
        return self.tokens.size


class VoiceCloner:
    """
    Voice cloning utility for extracting voice profiles from reference audio.
    
    This class provides methods to extract voice characteristics from audio files
    and save them as reusable voice profiles.
    """
    
    def __init__(self, model_path: str, audio_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the voice cloner.
        
        Args:
            model_path: Path to the Higgs Audio model
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=self.device)
        
    def extract_voice_profile(self, audio_path: str) -> VoiceProfile:
        """
        Extract voice profile from an audio file.
        
        Args:
            audio_path: Path to the reference audio file
            
        Returns:
            VoiceProfile object containing the extracted voice characteristics
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load and encode the audio
        import librosa
        audio_array, sample_rate = librosa.load(audio_path, sr=self.audio_tokenizer.sampling_rate)
        
        # Encode audio to tokens
        audio_tokens = self.audio_tokenizer.encode(audio_array, sample_rate)
        voice_tokens = audio_tokens[0, 0].detach().cpu().numpy()
        
        # Create metadata
        metadata = {
            "audio_path": audio_path,
            "sample_rate": sample_rate,
            "duration": len(audio_array) / sample_rate,
            "tokenizer_sampling_rate": self.audio_tokenizer.sampling_rate,
        }
        
        return VoiceProfile(voice_tokens, metadata)
    
    def extract_voice_profile_from_array(self, audio_array: np.ndarray, sample_rate: int) -> VoiceProfile:
        """
        Extract voice profile from a numpy array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            VoiceProfile object containing the extracted voice characteristics
        """
        # Encode audio to tokens
        audio_tokens = self.audio_tokenizer.encode(audio_array, sample_rate)
        voice_tokens = audio_tokens[0, 0].detach().cpu().numpy()
        
        # Create metadata
        metadata = {
            "sample_rate": sample_rate,
            "duration": len(audio_array) / sample_rate,
            "tokenizer_sampling_rate": self.audio_tokenizer.sampling_rate,
        }
        
        return VoiceProfile(voice_tokens, metadata)
    
    def save_voice_profile(self, voice_profile: VoiceProfile, output_path: str):
        """
        Save voice profile to a .npy file.
        
        Args:
            voice_profile: VoiceProfile object to save
            output_path: Path where to save the profile
        """
        # Save tokens and metadata
        profile_data = {
            "tokens": voice_profile.tokens,
            "metadata": voice_profile.metadata
        }
        np.save(output_path, profile_data)
    
    def load_voice_profile(self, profile_path: str) -> VoiceProfile:
        """
        Load voice profile from a .npy file.
        
        Args:
            profile_path: Path to the voice profile file
            
        Returns:
            VoiceProfile object
        """
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Voice profile file not found: {profile_path}")
        
        profile_data = np.load(profile_path, allow_pickle=True).item()
        return VoiceProfile(profile_data["tokens"], profile_data["metadata"])
    
    def get_voice_profile_info(self, voice_profile: VoiceProfile) -> Dict[str, Any]:
        """
        Get information about a voice profile.
        
        Args:
            voice_profile: VoiceProfile object
            
        Returns:
            Dictionary with profile information
        """
        info = {
            "shape": voice_profile.shape,
            "size": voice_profile.size,
            "metadata": voice_profile.metadata
        }
        return info


class TTSGenerator:
    """
    Text-to-Speech generator using voice profiles.
    
    This class provides methods to generate speech from text using pre-extracted
    voice profiles, supporting both single-shot and long-form generation.
    """
    
    def __init__(self, model_path: str, audio_tokenizer_path: str, device: str = "cuda", 
                 use_static_kv_cache: bool = True, kv_cache_lengths: List[int] = None):
        """
        Initialize the TTS generator.
        
        Args:
            model_path: Path to the Higgs Audio model
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
            use_static_kv_cache: Whether to use static KV cache for faster generation (GPU only)
            kv_cache_lengths: List of KV cache sizes for different sequence lengths
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Set default KV cache lengths if not provided
        if kv_cache_lengths is None:
            kv_cache_lengths = [1024, 4096, 8192]
        
        self.serve_engine = HiggsAudioServeEngine(
            model_path,
            audio_tokenizer_path,
            device=self.device,
            kv_cache_lengths=kv_cache_lengths
        )
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=self.device)
        self.collator = HiggsAudioSampleCollator()
        
        # Enable static KV cache if requested and on GPU
        if use_static_kv_cache and "cuda" in self.device:
            self._init_static_kv_cache()
        else:
            self.kv_caches = None

    def _init_static_kv_cache(self):
        """Initialize static KV cache for faster generation."""
        from transformers.cache_utils import StaticCache
        from copy import deepcopy
        
        cache_config = deepcopy(self.serve_engine._config.text_config)
        cache_config.num_hidden_layers = self.serve_engine._config.text_config.num_hidden_layers
        if self.serve_engine._config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.serve_engine._config.audio_dual_ffn_layers)
        
        # Create KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.serve_engine._model.device,
                dtype=self.serve_engine._model.dtype,
            )
            for length in sorted([1024, 4096, 8192])
        }
        
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self.device:
            self.serve_engine._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        """Prepare KV caches for generation."""
        if self.kv_caches:
            for kv_cache in self.kv_caches.values():
                kv_cache.reset()

    def generate_tts(self, text: str, voice_profile: VoiceProfile, **kwargs) -> HiggsAudioResponse:
        """
        Generate TTS audio using a voice profile (single-shot).
        
        Args:
            text: Text to convert to speech
            voice_profile: VoiceProfile object
            **kwargs: Additional generation parameters (temperature, top_p, etc.)
            
        Returns:
            HiggsAudioResponse containing the generated audio
        """
        # Create audio content from voice profile
        audio_content = self._create_audio_content_from_profile(voice_profile)
        
        # Prepare messages
        messages = []
        messages.append(Message(role="user", content=text))
        messages.append(Message(role="assistant", content=audio_content))
        
        # Create ChatMLSample
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Prepare KV caches if using static cache
        if self.kv_caches:
            self._prepare_kv_caches()

        # Generate audio
        response = self.serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            seed=kwargs.get("seed", None),
            past_key_values_buckets=self.kv_caches if self.kv_caches else None,
            ras_win_len=kwargs.get("ras_win_len", 7),
            ras_win_max_num_repeat=kwargs.get("ras_win_max_num_repeat", 2)
        )
        
        return response
    
    def generate_long_form_tts(self, text: str, voice_profile: VoiceProfile, **kwargs) -> HiggsAudioResponse:
        """
        Generate long-form TTS audio using a voice profile with seamless chunking.
        
        Args:
            text: Long text to convert to speech
            voice_profile: VoiceProfile object
            **kwargs: Additional generation parameters
            
        Returns:
            HiggsAudioResponse containing the generated audio
        """
        # Get chunking parameters
        chunk_method = kwargs.get("chunk_method", "word")
        chunk_max_word_num = kwargs.get("chunk_max_word_num", 200)
        chunk_max_num_turns = kwargs.get("chunk_max_num_turns", 1)
        generation_chunk_buffer_size = kwargs.get("generation_chunk_buffer_size", None)
        
        # Chunk the text
        chunked_text = self._prepare_chunk_text(
            text, 
            chunk_method=chunk_method, 
            chunk_max_word_num=chunk_max_word_num,
            chunk_max_num_turns=chunk_max_num_turns
        )
        
        # Convert voice profile to audio tokens for context
        voice_profile_tensor = torch.from_numpy(voice_profile.tokens).unsqueeze(0)
        voice_audio_tokens = voice_profile_tensor.to(self.device)
        
        # Initialize audio context with voice profile
        context_audio_ids = [voice_audio_tokens]
        generated_audio_ids = []
        all_generated_audio = []

        # Prepare KV caches if using static cache
        if self.kv_caches:
            self._prepare_kv_caches()

        # Process each chunk
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), 
            desc="Generating audio chunks", 
            total=len(chunked_text)
        ):
            # Add user message for this chunk
            generation_messages = [Message(role="user", content=chunk_text)]
            
            # Create ChatMLSample for this chunk
            chat_ml_sample = ChatMLSample(messages=generation_messages)
            
            # Prepare input tokens
            input_tokens, _, _, _ = prepare_chatml_sample(chat_ml_sample, self.serve_engine._tokenizer)
            postfix = self.serve_engine._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", 
                add_special_tokens=False
            )
            input_tokens.extend(postfix)
            
            # Prepare audio context (voice profile + previous generated audio)
            context_audio_ids_combined = context_audio_ids + generated_audio_ids
            
            # Create dataset sample
            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids_combined], dim=1)
                if context_audio_ids_combined
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids_combined], dtype=torch.long), dim=0
                )
                if context_audio_ids_combined
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )
            
            # Prepare batch
            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self.device)
            
            # Generate audio for this chunk
            outputs = self.serve_engine._model.generate(
                **batch,
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                use_cache=True,
                do_sample=True,
                temperature=kwargs.get("temperature", 0.3),
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 0.95),
                past_key_values_buckets=self.kv_caches if self.kv_caches else self.serve_engine.kv_caches,
                ras_win_len=kwargs.get("ras_win_len", 7),
                ras_win_max_num_repeat=kwargs.get("ras_win_max_num_repeat", 2),
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self.serve_engine._tokenizer,
                seed=kwargs.get("seed", None),
            )
            
            # Process generated audio tokens
            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self.serve_engine._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(
                    audio_out_ids.clip(0, self.audio_tokenizer.codebook_size - 1)[:, 1:-1]
                )
            
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            generated_audio_ids.append(audio_out_ids)
            
            # Decode audio tokens to waveform
            decoded_audio = self.audio_tokenizer.decode(audio_out_ids)
            chunk_audio = decoded_audio[0, 0].detach().cpu().numpy()
            all_generated_audio.append(chunk_audio)
            
            # Manage context buffer
            if generation_chunk_buffer_size and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
        
        # Concatenate all audio chunks
        final_audio = np.concatenate(all_generated_audio)
        
        # Create response object
        response = HiggsAudioResponse(
            audio=final_audio,
            sampling_rate=self.audio_tokenizer.sampling_rate,
            generated_text="",
            usage={"chunks": len(chunked_text)}
        )
        
        return response
    
    def save_audio(self, response: HiggsAudioResponse, output_path: str):
        """
        Save generated audio to a file.
        
        Args:
            response: HiggsAudioResponse object
            output_path: Path where to save the audio file
        """
        if response.audio is None:
            raise ValueError("No audio data in response")
        
        import soundfile as sf
        sf.write(output_path, response.audio, response.sampling_rate)
    
    def _create_audio_content_from_profile(self, voice_profile: VoiceProfile) -> AudioContent:
        """Create AudioContent from voice profile tokens."""
        voice_profile_tensor = torch.from_numpy(voice_profile.tokens).unsqueeze(0)
        decoded_audio = self.audio_tokenizer.decode(voice_profile_tensor)
        
        # Convert to base64 for AudioContent
        import base64
        import io
        
        audio_int16 = (decoded_audio[0, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return AudioContent(raw_audio=audio_base64, audio_url="placeholder")
    
    def _prepare_chunk_text(self, text, chunk_method=None, chunk_max_word_num=200, chunk_max_num_turns=1):
        """Prepare text chunks for long-form generation."""
        if chunk_method is None:
            return [text]
        elif chunk_method == "speaker":
            lines = text.split("\n")
            speaker_chunks = []
            speaker_utterance = ""
            for line in lines:
                line = line.strip()
                if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                    if speaker_utterance:
                        speaker_chunks.append(speaker_utterance.strip())
                    speaker_utterance = line
                else:
                    if speaker_utterance:
                        speaker_utterance += "\n" + line
                    else:
                        speaker_utterance = line
            if speaker_utterance:
                speaker_chunks.append(speaker_utterance.strip())
            if chunk_max_num_turns > 1:
                merged_chunks = []
                for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                    merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                    merged_chunks.append(merged_chunk)
                return merged_chunks
            return speaker_chunks
        elif chunk_method == "word":
            language = langid.classify(text)[0]
            paragraphs = text.split("\n\n")
            chunks = []
            for idx, paragraph in enumerate(paragraphs):
                if language == "zh":
                    words = list(jieba.cut(paragraph, cut_all=False))
                    for i in range(0, len(words), chunk_max_word_num):
                        chunk = "".join(words[i : i + chunk_max_word_num])
                        if chunk.strip():
                            chunks.append(chunk)
                else:
                    words = paragraph.split(" ")
                    for i in range(0, len(words), chunk_max_word_num):
                        chunk = " ".join(words[i : i + chunk_max_word_num])
                        if chunk.strip():
                            chunks.append(chunk)
            return chunks
        else:
            return [text] 