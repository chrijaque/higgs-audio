#!/usr/bin/env python3
"""
Command-line interface for TTS generation functionality.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from higgs_audio import TTSGenerator, load_voice_profile


def main():
    """Main CLI function for TTS generation."""
    parser = argparse.ArgumentParser(
        description="Generate TTS audio using voice profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-shot TTS generation
  higgs-tts --model_path /path/to/model --audio_tokenizer_path /path/to/tokenizer \\
           --voice_profile voice_profile.npy --text "Hello world" --output_audio output.wav
  
  # Long-form TTS generation with word-based chunking
  higgs-tts --model_path /path/to/model --audio_tokenizer_path /path/to/tokenizer \\
           --voice_profile voice_profile.npy --text "Long text..." \\
           --chunk_method word --output_audio long_output.wav
  
  # Long-form TTS generation with speaker-based chunking
  higgs-tts --model_path /path/to/model --audio_tokenizer_path /path/to/tokenizer \\
           --voice_profile voice_profile.npy --text "Speaker text..." \\
           --chunk_method speaker --output_audio speaker_output.wav
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
        help="Path where to save the generated audio file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on (default: cuda)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation (default: 0.3)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (default: None)"
    )
    
    # Long-form generation parameters
    parser.add_argument(
        "--chunk_method",
        type=str,
        choices=["word", "speaker", None],
        default=None,
        help="Method for chunking text (word, speaker, or None for single-shot)"
    )
    
    parser.add_argument(
        "--chunk_max_word_num",
        type=int,
        default=200,
        help="Maximum words per chunk for word-based chunking (default: 200)"
    )
    
    parser.add_argument(
        "--chunk_max_num_turns",
        type=int,
        default=1,
        help="Maximum turns per chunk for speaker-based chunking (default: 1)"
    )
    
    parser.add_argument(
        "--generation_chunk_buffer_size",
        type=int,
        default=None,
        help="Number of chunks to keep in buffer (default: None)"
    )
    
    parser.add_argument(
        "--ras_win_len",
        type=int,
        default=7,
        help="RAS window length for repetition control (default: 7)"
    )
    
    parser.add_argument(
        "--ras_win_max_num_repeat",
        type=int,
        default=2,
        help="Maximum RAS repetitions (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_tokenizer_path):
        print(f"Error: Audio tokenizer path does not exist: {args.audio_tokenizer_path}")
        sys.exit(1)
    
    if not os.path.exists(args.voice_profile):
        print(f"Error: Voice profile file does not exist: {args.voice_profile}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_audio)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize TTS generator
        print(f"Initializing TTS generator...")
        tts = TTSGenerator(
            model_path=args.model_path,
            audio_tokenizer_path=args.audio_tokenizer_path,
            device=args.device
        )
        
        # Load voice profile
        print(f"Loading voice profile from: {args.voice_profile}")
        voice_profile = load_voice_profile(args.voice_profile)
        
        # Generate TTS
        print(f"Generating TTS for text: {args.text[:50]}{'...' if len(args.text) > 50 else ''}")
        
        if args.chunk_method is None:
            # Single-shot generation
            print("Using single-shot generation...")
            response = tts.generate_tts(
                text=args.text,
                voice_profile=voice_profile,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed
            )
        else:
            # Long-form generation
            print(f"Using long-form generation with {args.chunk_method} chunking...")
            response = tts.generate_long_form_tts(
                text=args.text,
                voice_profile=voice_profile,
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
        print(f"Saving audio to: {args.output_audio}")
        tts.save_audio(response, args.output_audio)
        
        # Print generation info
        print(f"✅ TTS generation completed successfully!")
        if hasattr(response, 'usage') and response.usage:
            if 'chunks' in response.usage:
                print(f"   Generated {response.usage['chunks']} chunks")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 