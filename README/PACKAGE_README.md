# Higgs Audio Package

A clean Python package for voice cloning and text-to-speech generation using Higgs Audio V2.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install higgs-audio
```

## Quick Start

### Voice Cloning

```python
from higgs_audio import VoiceCloner

# Initialize voice cloner
cloner = VoiceCloner(
    model_path="/path/to/higgs_audio_model",
    audio_tokenizer_path="/path/to/audio_tokenizer",
    device="cuda"
)

# Extract voice profile from reference audio
voice_profile = cloner.extract_voice_profile("reference_audio.wav")

# Save voice profile for later use
cloner.save_voice_profile(voice_profile, "voice_profile.npy")
```

### Text-to-Speech Generation

```python
from higgs_audio import TTSGenerator, load_voice_profile

# Initialize TTS generator
tts = TTSGenerator(
    model_path="/path/to/higgs_audio_model",
    audio_tokenizer_path="/path/to/audio_tokenizer",
    device="cuda"
)

# Load voice profile
voice_profile = load_voice_profile("voice_profile.npy")

# Generate single-shot TTS
response = tts.generate_tts(
    text="Hello world!",
    voice_profile=voice_profile,
    temperature=0.3
)

# Save generated audio
tts.save_audio(response, "output.wav")

# Generate long-form TTS
response = tts.generate_long_form_tts(
    text="Long text for audiobook generation...",
    voice_profile=voice_profile,
    chunk_method="word",
    chunk_max_word_num=200
)

tts.save_audio(response, "long_output.wav")
```

## API Reference

### VoiceCloner

Voice cloning utility for extracting voice profiles from reference audio.

#### Methods

- `extract_voice_profile(audio_path: str) -> VoiceProfile`: Extract voice profile from audio file
- `extract_voice_profile_from_array(audio_array: np.ndarray, sample_rate: int) -> VoiceProfile`: Extract voice profile from numpy array
- `save_voice_profile(voice_profile: VoiceProfile, output_path: str)`: Save voice profile to .npy file
- `load_voice_profile(profile_path: str) -> VoiceProfile`: Load voice profile from .npy file
- `get_voice_profile_info(voice_profile: VoiceProfile) -> Dict[str, Any]`: Get profile information

### TTSGenerator

Text-to-Speech generator using voice profiles.

#### Methods

- `generate_tts(text: str, voice_profile: VoiceProfile, **kwargs) -> HiggsAudioResponse`: Single-shot TTS generation
- `generate_long_form_tts(text: str, voice_profile: VoiceProfile, **kwargs) -> HiggsAudioResponse`: Long-form TTS generation
- `save_audio(response: HiggsAudioResponse, output_path: str)`: Save generated audio to file

#### Generation Parameters

- `max_new_tokens`: Maximum tokens to generate (default: 1024)
- `temperature`: Generation temperature (default: 0.3)
- `top_p`: Top-p sampling (default: 0.95)
- `top_k`: Top-k sampling (default: 50)
- `seed`: Random seed (default: None)

#### Long-form Parameters

- `chunk_method`: Chunking method ("word", "speaker", or None)
- `chunk_max_word_num`: Max words per chunk (default: 200)
- `chunk_max_num_turns`: Max turns per chunk (default: 1)
- `generation_chunk_buffer_size`: Context buffer size (default: None)
- `ras_win_len`: RAS window length (default: 7)
- `ras_win_max_num_repeat`: Max RAS repetitions (default: 2)

### VoiceProfile

Represents a voice profile extracted from reference audio.

#### Properties

- `tokens`: Numpy array of voice tokens
- `metadata`: Dictionary of profile metadata
- `shape`: Token array shape
- `size`: Total number of tokens

## Command Line Interface

### Voice Cloning

```bash
# Extract voice profile
higgs-vc --model_path /path/to/model \
         --audio_tokenizer_path /path/to/tokenizer \
         --input_audio reference.wav \
         --output_profile voice_profile.npy

# With custom device
higgs-vc --model_path /path/to/model \
         --audio_tokenizer_path /path/to/tokenizer \
         --input_audio reference.wav \
         --output_profile voice_profile.npy \
         --device cpu
```

### TTS Generation

```bash
# Single-shot TTS
higgs-tts --model_path /path/to/model \
          --audio_tokenizer_path /path/to/tokenizer \
          --voice_profile voice_profile.npy \
          --text "Hello world" \
          --output_audio output.wav

# Long-form TTS with word chunking
higgs-tts --model_path /path/to/model \
          --audio_tokenizer_path /path/to/tokenizer \
          --voice_profile voice_profile.npy \
          --text "Long text..." \
          --chunk_method word \
          --output_audio long_output.wav

# Long-form TTS with speaker chunking
higgs-tts --model_path /path/to/model \
          --audio_tokenizer_path /path/to/tokenizer \
          --voice_profile voice_profile.npy \
          --text "Speaker text..." \
          --chunk_method speaker \
          --output_audio speaker_output.wav
```

## API Integration Example

```python
from higgs_audio import VoiceCloner, TTSGenerator

class HiggsAudioAPI:
    def __init__(self, model_path: str, audio_tokenizer_path: str):
        self.voice_cloner = VoiceCloner(model_path, audio_tokenizer_path)
        self.tts_generator = TTSGenerator(model_path, audio_tokenizer_path)
        self.voice_profiles = {}
    
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

# Usage
api = HiggsAudioAPI("/path/to/model", "/path/to/tokenizer")
api.extract_voice_profile("reference.wav", "user_voice")
result = api.generate_tts("Hello world", "user_voice", "output.wav")
```

## Features

- **Voice Cloning**: Extract voice profiles from reference audio
- **Single-shot TTS**: Generate short audio clips
- **Long-form TTS**: Generate long audio with seamless chunking
- **Multilingual Support**: Works with English, Spanish, Chinese, and more
- **Cross-lingual Voice Cloning**: Clone voice from one language and use in another
- **Automatic Language Detection**: Detects text language for appropriate processing
- **Voice Profile Management**: Save, load, and reuse voice profiles
- **Command Line Interface**: Easy-to-use CLI tools
- **Clean API**: Simple Python interface for integration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.45+
- CUDA (recommended for GPU acceleration)

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please visit the [GitHub repository](https://github.com/boson-ai/higgs-audio). 