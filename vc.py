#!/usr/bin/env python3
"""
Voice Profile Extractor for Higgs Audio V2

This script extracts voice profiles from reference audio files and saves them as .npy files
for later use in TTS generation. The voice profile contains the encoded audio tokens
that represent the voice characteristics.

Usage:
    python voice_profile_extractor.py --ref_audio path/to/audio.wav --output_profile voice_profile.npy
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


class VoiceProfileExtractor:
    """Extracts voice profiles from reference audio files."""
    
    def __init__(self, audio_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the voice profile extractor.
        
        Args:
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the tokenizer on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=self.device)
        
    def extract_voice_profile(self, audio_path: str) -> np.ndarray:
        """
        Extract voice profile from an audio file.
        
        Args:
            audio_path: Path to the reference audio file
            
        Returns:
            Voice profile as numpy array containing encoded audio tokens
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Encode the audio to get voice profile tokens
        voice_profile_tokens = self.audio_tokenizer.encode(audio_path)
        
        # Convert to numpy array and move to CPU
        voice_profile_np = voice_profile_tokens.squeeze(0).cpu().numpy()
        
        return voice_profile_np
    
    def extract_voice_profile_from_array(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract voice profile from an audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Voice profile as numpy array containing encoded audio tokens
        """
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Encode the audio
        voice_profile_tokens = self.audio_tokenizer.encode(audio_tensor, sample_rate)
        
        # Convert to numpy array and move to CPU
        voice_profile_np = voice_profile_tokens.squeeze(0).cpu().numpy()
        
        return voice_profile_np
    
    def save_voice_profile(self, voice_profile: np.ndarray, output_path: str):
        """
        Save voice profile to a .npy file.
        
        Args:
            voice_profile: Voice profile as numpy array
            output_path: Path to save the voice profile
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the voice profile
        np.save(output_path, voice_profile)
        print(f"Voice profile saved to: {output_path}")
        print(f"Voice profile shape: {voice_profile.shape}")
        
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
    
    def get_voice_profile_info(self, voice_profile: np.ndarray) -> dict:
        """
        Get information about a voice profile.
        
        Args:
            voice_profile: Voice profile as numpy array
            
        Returns:
            Dictionary containing voice profile information
        """
        return {
            "shape": voice_profile.shape,
            "dtype": str(voice_profile.dtype),
            "min_value": float(voice_profile.min()),
            "max_value": float(voice_profile.max()),
            "mean_value": float(voice_profile.mean()),
            "std_value": float(voice_profile.std()),
            "num_codebooks": voice_profile.shape[0] if len(voice_profile.shape) > 1 else 1,
            "sequence_length": voice_profile.shape[1] if len(voice_profile.shape) > 1 else voice_profile.shape[0]
        }


def main():
    parser = argparse.ArgumentParser(description="Extract voice profiles from reference audio files")
    parser.add_argument(
        "--ref_audio", 
        type=str, 
        required=True,
        help="Path to the reference audio file"
    )
    parser.add_argument(
        "--output_profile", 
        type=str, 
        required=True,
        help="Path to save the voice profile (.npy file)"
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
        help="Device to run the tokenizer on"
    )
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Print detailed information about the extracted voice profile"
    )
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = VoiceProfileExtractor(args.audio_tokenizer, device=args.device)
    
    # Extract voice profile
    print(f"Extracting voice profile from: {args.ref_audio}")
    voice_profile = extractor.extract_voice_profile(args.ref_audio)
    
    # Save voice profile
    extractor.save_voice_profile(voice_profile, args.output_profile)
    
    # Print information if requested
    if args.info:
        info = extractor.get_voice_profile_info(voice_profile)
        print("\nVoice Profile Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 