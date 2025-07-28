#!/usr/bin/env python3
"""
Example showing how to use Higgs Audio package in an API application.

This demonstrates the clean API that your application can use to integrate
voice cloning and TTS generation functionality.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from higgs_audio import VoiceCloner, TTSGenerator, load_voice_profile, save_voice_profile


def example_voice_cloning():
    """Example of voice cloning workflow."""
    print("=== Voice Cloning Example ===")
    
    # Initialize voice cloner
    cloner = VoiceCloner(
        model_path="/path/to/higgs_audio_model",
        audio_tokenizer_path="/path/to/audio_tokenizer",
        device="cuda"
    )
    
    # Extract voice profile from reference audio
    voice_profile = cloner.extract_voice_profile("examples/voice_prompts/belinda.wav")
    
    # Save voice profile for later use
    cloner.save_voice_profile(voice_profile, "belinda_voice_profile.npy")
    
    # Get information about the voice profile
    info = cloner.get_voice_profile_info(voice_profile)
    print(f"Voice profile shape: {info['shape']}")
    print(f"Voice profile size: {info['size']}")
    print(f"Metadata: {info['metadata']}")
    
    return voice_profile


def example_tts_generation(voice_profile):
    """Example of TTS generation workflow."""
    print("\n=== TTS Generation Example ===")
    
    # Initialize TTS generator
    tts = TTSGenerator(
        model_path="/path/to/higgs_audio_model",
        audio_tokenizer_path="/path/to/audio_tokenizer",
        device="cuda"
    )
    
    # Single-shot TTS generation
    print("Generating single-shot TTS...")
    response = tts.generate_tts(
        text="Hello world! This is a test of the Higgs Audio TTS system.",
        voice_profile=voice_profile,
        temperature=0.3,
        top_p=0.95
    )
    
    # Save the generated audio
    tts.save_audio(response, "single_shot_output.wav")
    print("Single-shot TTS saved to: single_shot_output.wav")
    
    # Long-form TTS generation
    print("\nGenerating long-form TTS...")
    long_text = """
    This is a longer text that demonstrates the long-form TTS capabilities of Higgs Audio.
    The system can handle much longer texts by automatically chunking them into smaller pieces
    and generating audio for each chunk while maintaining continuity between chunks.
    
    This is particularly useful for generating audiobooks, podcasts, or any other long-form
    audio content where you need to maintain consistent voice characteristics throughout
    the entire generation process.
    """
    
    response = tts.generate_long_form_tts(
        text=long_text,
        voice_profile=voice_profile,
        chunk_method="word",
        chunk_max_word_num=50,
        temperature=0.3,
        top_p=0.95
    )
    
    # Save the generated audio
    tts.save_audio(response, "long_form_output.wav")
    print("Long-form TTS saved to: long_form_output.wav")
    
    # Print generation info
    if hasattr(response, 'usage') and response.usage:
        if 'chunks' in response.usage:
            print(f"Generated {response.usage['chunks']} chunks")


def example_api_integration():
    """Example of how to integrate this into an API application."""
    print("\n=== API Integration Example ===")
    
    class HiggsAudioAPI:
        """Example API class that wraps Higgs Audio functionality."""
        
        def __init__(self, model_path: str, audio_tokenizer_path: str):
            self.voice_cloner = VoiceCloner(model_path, audio_tokenizer_path)
            self.tts_generator = TTSGenerator(model_path, audio_tokenizer_path)
            self.voice_profiles = {}  # Cache for voice profiles
        
        def extract_voice_profile(self, audio_path: str, profile_name: str):
            """Extract and cache a voice profile."""
            voice_profile = self.voice_cloner.extract_voice_profile(audio_path)
            self.voice_profiles[profile_name] = voice_profile
            return {"status": "success", "profile_name": profile_name}
        
        def generate_tts(self, text: str, profile_name: str, output_path: str, **kwargs):
            """Generate TTS using a cached voice profile."""
            if profile_name not in self.voice_profiles:
                return {"status": "error", "message": f"Voice profile '{profile_name}' not found"}
            
            voice_profile = self.voice_profiles[profile_name]
            response = self.tts_generator.generate_tts(text, voice_profile, **kwargs)
            self.tts_generator.save_audio(response, output_path)
            
            return {
                "status": "success",
                "output_path": output_path,
                "audio_length": len(response.audio) / response.sampling_rate if response.audio else 0
            }
        
        def generate_long_form_tts(self, text: str, profile_name: str, output_path: str, **kwargs):
            """Generate long-form TTS using a cached voice profile."""
            if profile_name not in self.voice_profiles:
                return {"status": "error", "message": f"Voice profile '{profile_name}' not found"}
            
            voice_profile = self.voice_profiles[profile_name]
            response = self.tts_generator.generate_long_form_tts(text, voice_profile, **kwargs)
            self.tts_generator.save_audio(response, output_path)
            
            return {
                "status": "success",
                "output_path": output_path,
                "audio_length": len(response.audio) / response.sampling_rate if response.audio else 0,
                "chunks": response.usage.get("chunks", 0) if hasattr(response, 'usage') else 0
            }
    
    # Example usage
    api = HiggsAudioAPI("/path/to/model", "/path/to/tokenizer")
    
    # Extract voice profile
    result = api.extract_voice_profile("examples/voice_prompts/belinda.wav", "belinda")
    print(f"Voice profile extraction: {result}")
    
    # Generate TTS
    result = api.generate_tts(
        text="Hello from the API!",
        profile_name="belinda",
        output_path="api_output.wav",
        temperature=0.3
    )
    print(f"TTS generation: {result}")
    
    # Generate long-form TTS
    result = api.generate_long_form_tts(
        text="This is a longer text for testing the API integration...",
        profile_name="belinda",
        output_path="api_long_output.wav",
        chunk_method="word"
    )
    print(f"Long-form TTS generation: {result}")


def main():
    """Main example function."""
    print("Higgs Audio Package API Example")
    print("=" * 50)
    
    # Note: These examples require actual model and tokenizer paths
    # Uncomment and modify the paths to run the examples
    
    # Example 1: Voice cloning
    # voice_profile = example_voice_cloning()
    
    # Example 2: TTS generation
    # example_tts_generation(voice_profile)
    
    # Example 3: API integration
    # example_api_integration()
    
    print("\nTo run these examples:")
    print("1. Update the model and tokenizer paths")
    print("2. Uncomment the function calls")
    print("3. Run: python examples/api_usage_example.py")


if __name__ == "__main__":
    main() 