#!/usr/bin/env python3
"""
Voice Profile TTS Generator for Higgs Audio V2

This script generates TTS audio using voice profiles extracted from reference audio files.
The voice profile contains the encoded audio tokens that represent the voice characteristics.

Supports both single-shot and long-form TTS generation with seamless chunk stitching.

Usage:
    python tts.py --voice_profile voice_profile.npy --text "Hello world" --output_audio output.wav
    python tts.py --voice_profile voice_profile.npy --text "Long text..." --chunk_method word --output_audio long_output.wav
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, List, Union
import tqdm
import langid
import jieba

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import prepare_chatml_sample, ChatMLDatasetSample
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from dataclasses import asdict


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 200, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces for long-form generation.
    
    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", or None.
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk when "word" chunking method is used.
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used.

    Returns
    -------
    List[str]
        The list of text chunks.
    """
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
        # For long-form generation, we will first divide the text into multiple paragraphs
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
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


class VoiceProfileTTSGenerator:
    """Generates TTS audio using voice profiles with support for long-form generation."""
    
    def __init__(self, model_path: str, audio_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the TTS generator.
        
        Args:
            model_path: Path to the Higgs Audio model
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.serve_engine = HiggsAudioServeEngine(
            model_path, 
            audio_tokenizer_path, 
            device=self.device
        )
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=self.device)
        
        # Initialize collator for long-form generation
        self.collator = HiggsAudioSampleCollator()
        
    def load_voice_profile(self, profile_path: str) -> np.ndarray:
        """
        Load voice profile from a .npy file.
        
        Args:
            profile_path: Path to the voice profile file
            
        Returns:
            Voice profile as numpy array
        """
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Voice profile file not found: {profile_path}")
            
        voice_profile = np.load(profile_path)
        return voice_profile
    
    def create_audio_content_from_profile(self, voice_profile: np.ndarray) -> AudioContent:
        """
        Create AudioContent from voice profile tokens.
        
        Args:
            voice_profile: Voice profile as numpy array
            
        Returns:
            AudioContent object for use in ChatMLSample
        """
        # Convert voice profile back to audio using the tokenizer
        voice_profile_tensor = torch.from_numpy(voice_profile).unsqueeze(0)
        
        # Decode the tokens back to audio
        decoded_audio = self.audio_tokenizer.decode(voice_profile_tensor)
        
        # Convert to base64 for AudioContent
        import base64
        import io
        
        # Convert to int16 and save to bytes
        audio_int16 = (decoded_audio[0, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return AudioContent(raw_audio=audio_base64, audio_url="placeholder")
    
    def generate_tts_with_voice_profile(
        self,
        text: str,
        voice_profile: np.ndarray,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: Optional[int] = None
    ) -> HiggsAudioResponse:
        """
        Generate TTS audio using a voice profile (single-shot).
        
        Args:
            text: Text to convert to speech
            voice_profile: Voice profile as numpy array
            system_prompt: Optional system prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            seed: Random seed for generation
            
        Returns:
            HiggsAudioResponse containing the generated audio
        """
        # Create audio content from voice profile
        audio_content = self.create_audio_content_from_profile(voice_profile)
        
        # Prepare messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        # Add user message with text
        messages.append(Message(role="user", content=text))
        
        # Add assistant message with voice profile audio
        messages.append(Message(role="assistant", content=audio_content))
        
        # Create ChatMLSample
        chat_ml_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        response = self.serve_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            seed=seed
        )
        
        return response
    
    def generate_long_form_tts_with_voice_profile(
        self,
        text: str,
        voice_profile: np.ndarray,
        system_prompt: Optional[str] = None,
        chunk_method: str = "word",
        chunk_max_word_num: int = 200,
        chunk_max_num_turns: int = 1,
        generation_chunk_buffer_size: Optional[int] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None
    ) -> HiggsAudioResponse:
        """
        Generate long-form TTS audio using a voice profile with seamless chunking.
        
        Args:
            text: Long text to convert to speech
            voice_profile: Voice profile as numpy array
            system_prompt: Optional system prompt
            chunk_method: Method for chunking ("word", "speaker", or None)
            chunk_max_word_num: Maximum words per chunk
            chunk_max_num_turns: Maximum turns per chunk for speaker method
            generation_chunk_buffer_size: Number of chunks to keep in buffer
            max_new_tokens: Maximum number of tokens to generate per chunk
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            ras_win_len: RAS window length for repetition control
            ras_win_max_num_repeat: Maximum RAS repetitions
            seed: Random seed for generation
            
        Returns:
            HiggsAudioResponse containing the generated audio
        """
        # Chunk the text
        chunked_text = prepare_chunk_text(
            text, 
            chunk_method=chunk_method, 
            chunk_max_word_num=chunk_max_word_num,
            chunk_max_num_turns=chunk_max_num_turns
        )
        
        print(f"Split text into {len(chunked_text)} chunks")
        
        # Convert voice profile to audio tokens for context
        voice_profile_tensor = torch.from_numpy(voice_profile).unsqueeze(0)
        voice_audio_tokens = voice_profile_tensor.to(self.device)
        
        # Prepare system and initial messages
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        
        # Initialize audio context with voice profile
        context_audio_ids = [voice_audio_tokens]
        generated_audio_ids = []
        all_generated_audio = []
        
        # Process each chunk
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), 
            desc="Generating audio chunks", 
            total=len(chunked_text)
        ):
            print(f"\nProcessing chunk {idx + 1}/{len(chunked_text)}: {chunk_text[:50]}...")
            
            # Add user message for this chunk
            generation_messages = messages + [
                Message(role="user", content=chunk_text)
            ]
            
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
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.serve_engine.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self.serve_engine._tokenizer,
                seed=seed,
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
            chunk_audio = decoded_audio[0, 0].cpu().numpy()
            all_generated_audio.append(chunk_audio)
            
            # Manage context buffer
            if generation_chunk_buffer_size and len(generated_audio_ids) > generation_chunk_buffer_size:
                # Keep only the most recent chunks in context
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
            
            # Add assistant message for next iteration
            messages.append(Message(role="user", content=chunk_text))
            messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
        
        # Concatenate all audio chunks
        final_audio = np.concatenate(all_generated_audio)
        
        # Create response object
        response = HiggsAudioResponse(
            audio=final_audio,
            generated_audio_tokens=torch.concat(generated_audio_ids, dim=1).cpu().numpy(),
            sampling_rate=24000,
            generated_text=text,
            usage={"total_chunks": len(chunked_text)}
        )
        
        return response
    
    def generate_tts_with_voice_profile_file(
        self,
        text: str,
        voice_profile_path: str,
        system_prompt: Optional[str] = None,
        chunk_method: Optional[str] = None,
        chunk_max_word_num: int = 200,
        chunk_max_num_turns: int = 1,
        generation_chunk_buffer_size: Optional[int] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None
    ) -> HiggsAudioResponse:
        """
        Generate TTS audio using a voice profile file.
        
        Args:
            text: Text to convert to speech
            voice_profile_path: Path to the voice profile file
            system_prompt: Optional system prompt
            chunk_method: Method for chunking ("word", "speaker", or None)
            chunk_max_word_num: Maximum words per chunk
            chunk_max_num_turns: Maximum turns per chunk for speaker method
            generation_chunk_buffer_size: Number of chunks to keep in buffer
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            ras_win_len: RAS window length for repetition control
            ras_win_max_num_repeat: Maximum RAS repetitions
            seed: Random seed for generation
            
        Returns:
            HiggsAudioResponse containing the generated audio
        """
        # Load voice profile
        voice_profile = self.load_voice_profile(voice_profile_path)
        
        # Choose generation method based on chunking
        if chunk_method is None:
            # Single-shot generation
            return self.generate_tts_with_voice_profile(
                text=text,
                voice_profile=voice_profile,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed
            )
        else:
            # Long-form generation
            return self.generate_long_form_tts_with_voice_profile(
                text=text,
                voice_profile=voice_profile,
                system_prompt=system_prompt,
                chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
                generation_chunk_buffer_size=generation_chunk_buffer_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed
            )
    
    def save_audio_response(self, response: HiggsAudioResponse, output_path: str):
        """
        Save audio response to file.
        
        Args:
            response: HiggsAudioResponse containing the generated audio
            output_path: Path to save the audio file
        """
        if response.audio is not None:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save audio
            torchaudio.save(
                output_path, 
                torch.from_numpy(response.audio)[None, :], 
                response.sampling_rate
            )
            print(f"Audio saved to: {output_path}")
            print(f"Audio length: {len(response.audio) / response.sampling_rate:.2f} seconds")
            
            if hasattr(response, 'usage') and response.usage:
                if 'total_chunks' in response.usage:
                    print(f"Generated from {response.usage['total_chunks']} chunks")
        else:
            print("No audio was generated")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS using voice profiles")
    parser.add_argument(
        "--voice_profile", 
        type=str, 
        required=True,
        help="Path to the voice profile file (.npy)"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--output_audio", 
        type=str, 
        required=True,
        help="Path to save the generated audio file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path to the Higgs Audio model"
    )
    parser.add_argument(
        "--audio_tokenizer", 
        type=str, 
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path to the Higgs Audio tokenizer"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default=None,
        help="Optional system prompt"
    )
    parser.add_argument(
        "--chunk_method",
        default=None,
        type=str,
        choices=[None, "speaker", "word"],
        help="The method to use for chunking the text. Options are 'speaker', 'word', or None for single-shot."
    )
    parser.add_argument(
        "--chunk_max_word_num",
        default=200,
        type=int,
        help="The maximum number of words for each chunk when 'word' chunking method is used."
    )
    parser.add_argument(
        "--chunk_max_num_turns",
        default=1,
        type=int,
        help="The maximum number of turns for each chunk when 'speaker' chunking method is used."
    )
    parser.add_argument(
        "--generation_chunk_buffer_size",
        default=None,
        type=int,
        help="The maximal number of chunks to keep in the buffer for context."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate per chunk"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.3,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--ras_win_len",
        type=int,
        default=7,
        help="The window length for RAS sampling."
    )
    parser.add_argument(
        "--ras_win_max_num_repeat",
        type=int,
        default=2,
        help="The maximum number of times to repeat the RAS window."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for generation"
    )
    
    args = parser.parse_args()
    
    # Initialize the TTS generator
    generator = VoiceProfileTTSGenerator(
        args.model_path, 
        args.audio_tokenizer, 
        device=args.device
    )
    
    # Generate TTS
    print(f"Generating TTS for text: {args.text[:100]}...")
    print(f"Using voice profile: {args.voice_profile}")
    if args.chunk_method:
        print(f"Using chunking method: {args.chunk_method}")
    
    response = generator.generate_tts_with_voice_profile_file(
        text=args.text,
        voice_profile_path=args.voice_profile,
        system_prompt=args.system_prompt,
        chunk_method=args.chunk_method,
        chunk_max_word_num=args.chunk_max_word_num,
        chunk_max_num_turns=args.chunk_max_num_turns,
        generation_chunk_buffer_size=args.generation_chunk_buffer_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ras_win_len=args.ras_win_len,
        ras_win_max_num_repeat=args.ras_win_max_num_repeat,
        seed=args.seed
    )
    
    # Save audio
    generator.save_audio_response(response, args.output_audio)
    
    # Print generation info
    if response.generated_text:
        print(f"Generated text length: {len(response.generated_text)} characters")
    
    if response.usage:
        print(f"Token usage: {response.usage}")


if __name__ == "__main__":
    main() 