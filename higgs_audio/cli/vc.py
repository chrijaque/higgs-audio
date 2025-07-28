#!/usr/bin/env python3
"""
Command-line interface for voice cloning functionality.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from higgs_audio import VoiceCloner


def main():
    """Main CLI function for voice cloning."""
    parser = argparse.ArgumentParser(
        description="Extract voice profiles from reference audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract voice profile from audio file
  higgs-vc --model_path /path/to/model --audio_tokenizer_path /path/to/tokenizer \\
           --input_audio reference.wav --output_profile voice_profile.npy
  
  # Extract voice profile with custom device
  higgs-vc --model_path /path/to/model --audio_tokenizer_path /path/to/tokenizer \\
           --input_audio reference.wav --output_profile voice_profile.npy --device cpu
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Higgs Audio model"
    )
    
    parser.add_argument(
        "--audio_tokenizer_path", 
        type=str,
        required=True,
        help="Path to the Higgs Audio tokenizer"
    )
    
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to the input audio file"
    )
    
    parser.add_argument(
        "--output_profile",
        type=str,
        required=True,
        help="Path where to save the voice profile (.npy file)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on (default: cuda)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about the extracted voice profile"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_tokenizer_path):
        print(f"Error: Audio tokenizer path does not exist: {args.audio_tokenizer_path}")
        sys.exit(1)
    
    if not os.path.exists(args.input_audio):
        print(f"Error: Input audio file does not exist: {args.input_audio}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_profile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize voice cloner
        print(f"Initializing voice cloner...")
        cloner = VoiceCloner(
            model_path=args.model_path,
            audio_tokenizer_path=args.audio_tokenizer_path,
            device=args.device
        )
        
        # Extract voice profile
        print(f"Extracting voice profile from: {args.input_audio}")
        voice_profile = cloner.extract_voice_profile(args.input_audio)
        
        # Save voice profile
        print(f"Saving voice profile to: {args.output_profile}")
        cloner.save_voice_profile(voice_profile, args.output_profile)
        
        # Print information if requested
        if args.info:
            info = cloner.get_voice_profile_info(voice_profile)
            print("\nVoice Profile Information:")
            print(f"  Shape: {info['shape']}")
            print(f"  Size: {info['size']}")
            print(f"  Metadata: {info['metadata']}")
        
        print(f"✅ Voice profile extracted successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 