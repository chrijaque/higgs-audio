#!/usr/bin/env python3
"""
Voice Profile Example for Higgs Audio V2

This script demonstrates the complete workflow:
1. Extract voice profiles from reference audio files
2. Use voice profiles for TTS generation
3. Compare results with original voice cloning

Usage:
    python voice_profile_example.py --ref_audio examples/voice_prompts/belinda.wav --text "Hello world"
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vc import VoiceProfileExtractor
from tts import VoiceProfileTTSGenerator


class VoiceProfileExample:
    """Complete example demonstrating voice profile extraction and TTS generation."""
    
    def __init__(self, model_path: str, audio_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the voice profile example.
        
        Args:
            model_path: Path to the Higgs Audio model
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the models on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_path = model_path
        self.audio_tokenizer_path = audio_tokenizer_path
        
        # Initialize components
        self.extractor = VoiceProfileExtractor(audio_tokenizer_path, device=self.device)
        self.tts_generator = VoiceProfileTTSGenerator(model_path, audio_tokenizer_path, device=self.device)
        
    def extract_and_save_voice_profile(self, audio_path: str, output_profile_path: str) -> np.ndarray:
        """
        Extract voice profile from audio and save it.
        
        Args:
            audio_path: Path to the reference audio file
            output_profile_path: Path to save the voice profile
            
        Returns:
            Voice profile as numpy array
        """
        print(f"Extracting voice profile from: {audio_path}")
        
        # Extract voice profile
        voice_profile = self.extractor.extract_voice_profile(audio_path)
        
        # Save voice profile
        self.extractor.save_voice_profile(voice_profile, output_profile_path)
        
        # Print voice profile information
        info = self.extractor.get_voice_profile_info(voice_profile)
        print("Voice Profile Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        return voice_profile
    
    def generate_tts_with_profile(self, text: str, voice_profile_path: str, output_audio_path: str):
        """
        Generate TTS using a voice profile.
        
        Args:
            text: Text to convert to speech
            voice_profile_path: Path to the voice profile file
            output_audio_path: Path to save the generated audio
        """
        print(f"Generating TTS with voice profile: {voice_profile_path}")
        print(f"Text: {text}")
        
        # Generate TTS
        response = self.tts_generator.generate_tts_with_voice_profile_file(
            text=text,
            voice_profile_path=voice_profile_path,
            temperature=0.3,
            max_new_tokens=1024
        )
        
        # Save audio
        self.tts_generator.save_audio_response(response, output_audio_path)
        
        return response
    
    def compare_with_original_voice_cloning(self, ref_audio_path: str, text: str, output_dir: str):
        """
        Compare voice profile TTS with original voice cloning method.
        
        Args:
            ref_audio_path: Path to the reference audio file
            text: Text to convert to speech
            output_dir: Directory to save comparison results
        """
        print("\n" + "="*60)
        print("COMPARISON: Voice Profile TTS vs Original Voice Cloning")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Method 1: Voice Profile TTS
        print("\n1. Voice Profile TTS Method:")
        print("-" * 30)
        
        # Extract voice profile
        profile_path = os.path.join(output_dir, "voice_profile.npy")
        voice_profile = self.extract_and_save_voice_profile(ref_audio_path, profile_path)
        
        # Generate TTS with voice profile
        profile_output_path = os.path.join(output_dir, "tts_with_profile.wav")
        profile_response = self.generate_tts_with_profile(text, profile_path, profile_output_path)
        
        # Method 2: Original Voice Cloning (using the generation.py approach)
        print("\n2. Original Voice Cloning Method:")
        print("-" * 30)
        
        # Import the generation script components
        from examples.generation import HiggsAudioModelClient, prepare_generation_context
        from boson_multimodal.data_types import Message, AudioContent
        from boson_multimodal.dataset.chatml_dataset import ChatMLSample
        
        # Initialize model client
        model_client = HiggsAudioModelClient(
            model_path=self.model_path,
            audio_tokenizer=self.audio_tokenizer_path,
            device_id=0 if self.device == "cuda" else None
        )
        
        # Prepare context for original voice cloning
        audio_tokenizer = self.extractor.audio_tokenizer
        messages, audio_ids, _ = prepare_generation_context(
            scene_prompt=None,
            ref_audio=os.path.basename(ref_audio_path).replace('.wav', ''),
            ref_audio_in_system_message=False,
            audio_tokenizer=audio_tokenizer,
            speaker_tags=[]
        )
        
        # Generate with original method
        original_output_path = os.path.join(output_dir, "tts_original_cloning.wav")
        concat_wv, sr, text_result = model_client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=[text],
            generation_chunk_buffer_size=None,
            temperature=0.3,
            seed=123
        )
        
        # Save original method result
        torchaudio.save(original_output_path, torch.from_numpy(concat_wv)[None, :], sr)
        print(f"Original voice cloning audio saved to: {original_output_path}")
        print(f"Audio length: {len(concat_wv) / sr:.2f} seconds")
        
        # Compare results
        print("\n3. Comparison Results:")
        print("-" * 30)
        print(f"Voice Profile TTS: {profile_output_path}")
        print(f"Original Voice Cloning: {original_output_path}")
        print(f"Voice Profile Size: {os.path.getsize(profile_path) / 1024:.1f} KB")
        print(f"Voice Profile Shape: {voice_profile.shape}")
        
        return {
            "profile_path": profile_path,
            "profile_audio": profile_output_path,
            "original_audio": original_output_path,
            "voice_profile_info": self.extractor.get_voice_profile_info(voice_profile)
        }
    
    def batch_extract_profiles(self, audio_dir: str, output_dir: str):
        """
        Extract voice profiles from all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save voice profiles
        """
        print(f"Batch extracting voice profiles from: {audio_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
        
        profiles = {}
        
        for audio_file in audio_files:
            print(f"\nProcessing: {audio_file.name}")
            
            # Extract voice profile
            profile_name = audio_file.stem + ".npy"
            profile_path = os.path.join(output_dir, profile_name)
            
            try:
                voice_profile = self.extract_and_save_voice_profile(str(audio_file), profile_path)
                profiles[audio_file.name] = {
                    "profile_path": profile_path,
                    "voice_profile": voice_profile,
                    "info": self.extractor.get_voice_profile_info(voice_profile)
                }
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
        
        print(f"\nExtracted {len(profiles)} voice profiles to: {output_dir}")
        return profiles
    
    def batch_generate_tts(self, profiles_dir: str, text: str, output_dir: str):
        """
        Generate TTS for all voice profiles in a directory.
        
        Args:
            profiles_dir: Directory containing voice profile files
            text: Text to convert to speech
            output_dir: Directory to save generated audio files
        """
        print(f"Batch generating TTS for profiles in: {profiles_dir}")
        print(f"Text: {text}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all profile files
        profile_files = list(Path(profiles_dir).glob("*.npy"))
        
        results = {}
        
        for profile_file in profile_files:
            print(f"\nGenerating TTS with profile: {profile_file.name}")
            
            # Generate TTS
            output_audio_path = os.path.join(output_dir, f"{profile_file.stem}_tts.wav")
            
            try:
                response = self.generate_tts_with_profile(text, str(profile_file), output_audio_path)
                results[profile_file.name] = {
                    "output_audio": output_audio_path,
                    "response": response
                }
            except Exception as e:
                print(f"Error generating TTS for {profile_file.name}: {e}")
        
        print(f"\nGenerated TTS for {len(results)} voice profiles in: {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Voice Profile Example for Higgs Audio V2")
    parser.add_argument(
        "--ref_audio", 
        type=str, 
        required=True,
        help="Path to the reference audio file"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default="Hello world! This is a test of voice profile TTS generation.",
        help="Text to convert to speech"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="voice_profile_results",
        help="Directory to save results"
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
        help="Device to run the models on"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="compare",
        choices=["extract", "generate", "compare", "batch_extract", "batch_generate"],
        help="Mode to run the example in"
    )
    parser.add_argument(
        "--voice_profile", 
        type=str, 
        default=None,
        help="Path to voice profile file (for generate mode)"
    )
    parser.add_argument(
        "--profiles_dir", 
        type=str, 
        default=None,
        help="Directory containing voice profiles (for batch modes)"
    )
    
    args = parser.parse_args()
    
    # Initialize the example
    example = VoiceProfileExample(args.model_path, args.audio_tokenizer, device=args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "extract":
        # Extract voice profile only
        profile_path = os.path.join(args.output_dir, "voice_profile.npy")
        example.extract_and_save_voice_profile(args.ref_audio, profile_path)
        
    elif args.mode == "generate":
        # Generate TTS with voice profile
        if not args.voice_profile:
            raise ValueError("--voice_profile is required for generate mode")
        
        output_audio_path = os.path.join(args.output_dir, "generated_tts.wav")
        example.generate_tts_with_profile(args.text, args.voice_profile, output_audio_path)
        
    elif args.mode == "compare":
        # Compare voice profile TTS with original voice cloning
        example.compare_with_original_voice_cloning(args.ref_audio, args.text, args.output_dir)
        
    elif args.mode == "batch_extract":
        # Batch extract voice profiles
        if not args.profiles_dir:
            args.profiles_dir = os.path.join(args.output_dir, "profiles")
        
        example.batch_extract_profiles(args.ref_audio, args.profiles_dir)
        
    elif args.mode == "batch_generate":
        # Batch generate TTS
        if not args.profiles_dir:
            raise ValueError("--profiles_dir is required for batch_generate mode")
        
        output_audio_dir = os.path.join(args.output_dir, "generated_audio")
        example.batch_generate_tts(args.profiles_dir, args.text, output_audio_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 