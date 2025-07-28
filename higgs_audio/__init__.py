"""
Higgs Audio V2 - A powerful audio foundation model for expressive audio generation.

This package provides easy-to-use interfaces for voice cloning, text-to-speech generation,
and voice profile management.

Example usage:
    from higgs_audio import VoiceCloner, TTSGenerator
    
    # Voice cloning
    cloner = VoiceCloner(model_path="path/to/model", audio_tokenizer_path="path/to/tokenizer")
    voice_profile = cloner.extract_voice_profile("reference_audio.wav")
    cloner.save_voice_profile(voice_profile, "voice_profile.npy")
    
    # TTS generation
    tts = TTSGenerator(model_path="path/to/model", audio_tokenizer_path="path/to/tokenizer")
    audio = tts.generate_tts("Hello world", voice_profile)
    tts.save_audio(audio, "output.wav")
"""

from .core import VoiceCloner, TTSGenerator, VoiceProfile
from .utils import load_voice_profile, save_voice_profile, get_voice_profile_info

__version__ = "0.1.0"
__author__ = "Boson AI"
__email__ = "info@boson.ai"

__all__ = [
    "VoiceCloner",
    "TTSGenerator", 
    "VoiceProfile",
    "load_voice_profile",
    "save_voice_profile",
    "get_voice_profile_info",
] 