#!/usr/bin/env python3
"""
Voice Profile Extractor for Higgs Audio V2

This script extracts voice profiles from reference audio files and saves them as .npy files
for later use in TTS generation. The voice profile contains the encoded audio tokens
that represent the voice characteristics.

Usage:
    python vc.py --ref_audio path/to/audio.wav --output_profile voice_profile.npy
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


class FixedHiggsAudioTokenizer:
    """
    Wrapper around HiggsAudioTokenizer that fixes the gradient issue in decode method.
    This provides a safe interface that handles the bug automatically.
    """
    
    def __init__(self, tokenizer_name_or_path, device="cuda"):
        """
        Initialize the fixed tokenizer wrapper.
        
        Args:
            tokenizer_name_or_path: Path to the tokenizer
            device: Device to run on ('cuda' or 'cpu')
        """
        # Follow official pattern: use CPU for audio tokenizer on MPS
        audio_tokenizer_device = "cpu" if device == "mps" else device
        self.tokenizer = load_higgs_audio_tokenizer(tokenizer_name_or_path, device=audio_tokenizer_device)
        self.device = device
    
    def decode(self, vq_code: torch.Tensor) -> np.ndarray:
        """
        Fixed decode method that properly handles gradients.
        
        Args:
            vq_code: Voice profile tensor
            
        Returns:
            Decoded audio as numpy array
        """
        # Ensure input is detached to prevent gradient issues
        if vq_code.requires_grad:
            vq_code = vq_code.detach()
        
        # Use the tokenizer's decode method (which should now be fixed)
        try:
            return self.tokenizer.decode(vq_code)
        except RuntimeError as e:
            if "grad" in str(e).lower():
                # Fallback: manually implement the fixed decode
                return self._manual_decode(vq_code)
            else:
                raise e
    
    def _manual_decode(self, vq_code: torch.Tensor) -> np.ndarray:
        """
        Manual decode implementation as fallback if the tokenizer still has issues.
        """
        if self.tokenizer.quantizer_type == "RVQ":
            vq_code = vq_code.permute(1, 0, 2)
            quantized = self.tokenizer.quantizer.decode(vq_code)
            quantized = quantized.transpose(1, 2)
        else:
            vq_code = vq_code.permute(0, 2, 1)
            quantized = self.tokenizer.quantizer.get_output_from_indices(vq_code)
        
        quantized_acoustic = self.tokenizer.fc_post2(quantized).transpose(1, 2)
        o = self.tokenizer.decoder_2(quantized_acoustic)
        
        # Ensure proper gradient handling
        return o.detach().cpu().numpy()
    
    def encode(self, audio_path_or_wv, sr=None, **kwargs):
        """Delegate encode to the original tokenizer."""
        return self.tokenizer.encode(audio_path_or_wv, sr, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original tokenizer."""
        return getattr(self.tokenizer, name)


class VoiceProfileExtractor:
    """Extracts voice profiles from reference audio files following official Higgs Audio patterns."""
    
    def __init__(self, audio_tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the voice profile extractor.
        
        Args:
            audio_tokenizer_path: Path to the Higgs Audio tokenizer
            device: Device to run the tokenizer on ('cuda', 'cpu', or 'mps')
        """
        # Follow official device selection pattern
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"✅ Voice Profile Extractor initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Tokenizer: {audio_tokenizer_path}")
        print(f"   - Memory usage: ~1GB (lightweight tokenizer only)")
        
        # Use the fixed tokenizer wrapper
        self.audio_tokenizer = FixedHiggsAudioTokenizer(audio_tokenizer_path, device=self.device)
        
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
            
        print(f"🔍 Extracting voice profile from: {audio_path}")
        
        # Encode the audio to get voice profile tokens
        voice_profile_tokens = self.audio_tokenizer.encode(audio_path)
        
        # Convert to numpy array and move to CPU
        voice_profile_np = voice_profile_tokens.squeeze(0).detach().cpu().numpy()
        
        print(f"✅ Voice profile extracted: shape={voice_profile_np.shape}")
        
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
        print(f"🔍 Extracting voice profile from audio array: shape={audio_array.shape}, sr={sample_rate}")
        
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
        voice_profile_np = voice_profile_tokens.squeeze(0).detach().cpu().numpy()
        
        print(f"✅ Voice profile extracted: shape={voice_profile_np.shape}")
        
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
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ Voice profile saved to: {output_path}")
        print(f"   - Shape: {voice_profile.shape}")
        print(f"   - Size: {file_size_mb:.2f} MB")
        
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
        print(f"✅ Voice profile loaded: {profile_path}")
        print(f"   - Shape: {voice_profile.shape}")
        
        return voice_profile
    
    def get_voice_profile_info(self, voice_profile: np.ndarray) -> dict:
        """
        Get information about a voice profile.
        
        Args:
            voice_profile: Voice profile as numpy array
            
        Returns:
            Dictionary containing voice profile information
        """
        info = {
            "shape": voice_profile.shape,
            "dtype": str(voice_profile.dtype),
            "min_value": float(voice_profile.min()),
            "max_value": float(voice_profile.max()),
            "mean_value": float(voice_profile.mean()),
            "std_value": float(voice_profile.std()),
            "num_codebooks": voice_profile.shape[0] if len(voice_profile.shape) > 1 else 1,
            "sequence_length": voice_profile.shape[1] if len(voice_profile.shape) > 1 else voice_profile.shape[0]
        }
        
        # Calculate size in MB
        size_bytes = voice_profile.nbytes
        info["size_mb"] = size_bytes / (1024 * 1024)
        
        return info
    
    def decode_voice_profile_to_audio(self, voice_profile: np.ndarray) -> np.ndarray:
        """
        Decode voice profile back to audio using only the tokenizer (lightweight).
        
        Args:
            voice_profile: Voice profile as numpy array
            
        Returns:
            Decoded audio as numpy array
        """
        print(f"🔍 Decoding voice profile to audio: shape={voice_profile.shape}")
        
        # Convert to tensor
        voice_profile_tensor = torch.from_numpy(voice_profile).unsqueeze(0)
        
        # Use the lightweight tokenizer decode method
        decoded_audio = self.audio_tokenizer.decode(voice_profile_tensor)
        
        print(f"✅ Audio decoded: shape={decoded_audio.shape}")
        
        return decoded_audio
    
    def validate_voice_profile(self, voice_profile: np.ndarray) -> bool:
        """
        Validate that a voice profile has the correct format.
        
        Args:
            voice_profile: Voice profile as numpy array
            
        Returns:
            True if valid, False otherwise
        """
        # Check shape (should be 2D: [num_codebooks, sequence_length])
        if len(voice_profile.shape) != 2:
            print(f"❌ Invalid shape: expected 2D, got {len(voice_profile.shape)}D")
            return False
        
        # Check number of codebooks (should be 8 for Higgs Audio)
        if voice_profile.shape[0] != 8:
            print(f"❌ Invalid codebooks: expected 8, got {voice_profile.shape[0]}")
            return False
        
        # Check data type (should be integer indices)
        if not np.issubdtype(voice_profile.dtype, np.integer):
            print(f"❌ Invalid dtype: expected integer, got {voice_profile.dtype}")
            return False
        
        # Check value range (should be 0-1023 for codebook indices)
        if voice_profile.min() < 0 or voice_profile.max() >= 1024:
            print(f"❌ Invalid values: expected 0-1023, got {voice_profile.min()}-{voice_profile.max()}")
            return False
        
        print(f"✅ Voice profile validation passed: shape={voice_profile.shape}")
        return True


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
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run the tokenizer on ('auto' picks best available)"
    )
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Print detailed information about the extracted voice profile"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate the extracted voice profile"
    )
    parser.add_argument(
        "--example", 
        action="store_true",
        help="Show example of how to reuse the voice profile"
    )
    
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = VoiceProfileExtractor(args.audio_tokenizer, device=args.device)
    
    # Extract voice profile
    voice_profile = extractor.extract_voice_profile(args.ref_audio)
    
    # Validate if requested
    if args.validate:
        if not extractor.validate_voice_profile(voice_profile):
            print("❌ Voice profile validation failed!")
            return 1
    
    # Save voice profile
    extractor.save_voice_profile(voice_profile, args.output_profile)
    
    # Print information if requested
    if args.info:
        info = extractor.get_voice_profile_info(voice_profile)
        print("\nVoice Profile Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Show reuse example if requested
    if args.example:
        print("\n" + "="*60)
        print("EXAMPLE: How to Reuse This Voice Profile")
        print("="*60)
        
        print("\n1. Load the voice profile:")
        print(f"   voice_profile = np.load('{args.output_profile}')")
        
        print("\n2. Use with TTS generation:")
        print("   from tts import OfficialTTSGenerator")
        print("   tts_generator = OfficialTTSGenerator(")
        print("       model_path='bosonai/higgs-audio-v2-generation-3B-base',")
        print("       audio_tokenizer_path='bosonai/higgs-audio-v2-tokenizer'")
        print("   )")
        print("   response = tts_generator.generate_tts_with_voice_profile(")
        print("       text='Hello world',")
        print("       voice_profile=voice_profile")
        print("   )")
        
        print("\n3. Or use with command line:")
        print(f"   python tts.py --voice_profile {args.output_profile} --text 'Hello world'")
        
        print("\n4. Programmatic reuse example:")
        print("   # Load saved voice profile")
        print(f"   loaded_profile = extractor.load_voice_profile('{args.output_profile}')")
        print("   ")
        print("   # Generate TTS with the same voice")
        print("   response = tts_generator.generate_tts_with_voice_profile(")
        print("       text='This is the same voice speaking again!',")
        print("       voice_profile=loaded_profile")
        print("   )")
    
    print(f"✅ Voice profile extraction completed successfully!")


# Example usage functions
def extract_and_save_voice_profile(audio_path: str, output_path: str, device: str = "auto"):
    """
    Extract and save a voice profile for reuse.
    
    Args:
        audio_path: Path to reference audio file
        output_path: Path to save the voice profile
        device: Device to use ('auto', 'cuda', 'cpu', 'mps')
    
    Returns:
        Path to the saved voice profile
    """
    print(f"🔍 Extracting voice profile from: {audio_path}")
    
    # Initialize extractor
    extractor = VoiceProfileExtractor(
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        device=device
    )
    
    # Extract voice profile
    voice_profile = extractor.extract_voice_profile(audio_path)
    
    # Validate
    if not extractor.validate_voice_profile(voice_profile):
        raise ValueError("Voice profile validation failed!")
    
    # Save for reuse
    extractor.save_voice_profile(voice_profile, output_path)
    
    print(f"✅ Voice profile saved for reuse: {output_path}")
    return output_path


def load_and_use_voice_profile(profile_path: str, text: str, output_audio_path: str):
    """
    Load a saved voice profile and use it for TTS generation.
    
    Args:
        profile_path: Path to the saved voice profile
        text: Text to convert to speech
        output_audio_path: Path to save the generated audio
    """
    print(f"🔍 Loading voice profile: {profile_path}")
    
    # Load the voice profile
    extractor = VoiceProfileExtractor(
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        device="auto"
    )
    
    voice_profile = extractor.load_voice_profile(profile_path)
    
    # Generate TTS using the voice profile
    from tts import OfficialTTSGenerator
    
    tts_generator = OfficialTTSGenerator(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        device="cuda"
    )
    
    response = tts_generator.generate_tts_with_voice_profile(
        text=text,
        voice_profile=voice_profile,
        temperature=0.3
    )
    
    # Save the generated audio
    tts_generator.save_audio_response(response, output_audio_path)
    
    print(f"✅ TTS generated using voice profile: {output_audio_path}")


if __name__ == "__main__":
    main() 