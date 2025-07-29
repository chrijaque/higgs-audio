# Voice Profile System for Higgs Audio V2

This system allows you to extract voice profiles from reference audio files and use them for Text-to-Speech (TTS) generation, providing a more efficient and reusable approach to voice cloning.

## Overview

The voice profile system consists of three main components:

1. **`vc.py`** - Voice profile extraction from reference audio
2. **`tts.py`** - TTS generation using voice profiles (supports both single-shot and long-form)
3. **`voice_profile_example.py`** - Complete workflow demonstration

## Benefits

- **Efficiency**: Extract voice profile once, use it multiple times
- **Consistency**: Same voice characteristics across different generations
- **Speed**: No need to re-encode reference audio for each generation
- **Long-form Support**: Seamless chunking for extended content
- **Reusability**: Voice profiles can be shared and reused

## Installation

```bash
# Install required dependencies
pip install torch torchaudio numpy tqdm langid jieba

# Clone the repository (if not already done)
git clone <repository-url>
cd higgs-audio
```

## Quick Start

### 1. Extract a Voice Profile

```bash
python vc.py --audio_file examples/voice_prompts/belinda.wav --output_profile voice_profiles/belinda_profile.npy
```

### 2. Generate TTS (Single-shot)

```bash
python tts.py --voice_profile voice_profiles/belinda_profile.npy --text "Hello, this is a test." --output_audio output.wav
```

### 3. Generate Long-form TTS

```bash
python tts.py --voice_profile voice_profiles/belinda_profile.npy --text "Long text content..." --chunk_method word --output_audio long_output.wav
```

## Detailed Usage

### Voice Profile Extraction (`vc.py`)

#### Basic Usage

```bash
python vc.py --audio_file <input_audio.wav> --output_profile <output_profile.npy>
```

#### Advanced Options

```bash
python vc.py \
    --audio_file examples/voice_prompts/belinda.wav \
    --output_profile voice_profiles/belinda_profile.npy \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer bosonai/higgs-audio-v2-tokenizer \
    --device cuda
```

#### Programmatic Usage

```python
from vc import VoiceProfileExtractor

# Initialize extractor
extractor = VoiceProfileExtractor(
    audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
    device="cuda"
)

# Extract voice profile
voice_profile = extractor.extract_voice_profile("input_audio.wav")

# Save profile
extractor.save_voice_profile(voice_profile, "voice_profile.npy")

# Get profile information
info = extractor.get_voice_profile_info(voice_profile)
print(f"Profile shape: {info['shape']}")
print(f"Profile size: {info['size_mb']:.2f} MB")
```

### TTS Generation (`tts.py`)

#### Single-shot TTS

For short texts (recommended for < 1000 characters):

```bash
python tts.py \
    --voice_profile voice_profiles/belinda_profile.npy \
    --text "Hello, this is a short test message." \
    --output_audio short_output.wav \
    --temperature 0.3 \
    --max_new_tokens 512
```

#### Long-form TTS

For longer texts with seamless chunking:

```bash
python tts.py \
    --voice_profile voice_profiles/belinda_profile.npy \
    --text "Very long text content..." \
    --chunk_method word \
    --chunk_max_word_num 200 \
    --generation_chunk_buffer_size 5 \
    --output_audio long_output.wav \
    --temperature 0.3 \
    --max_new_tokens 1024 \
    --ras_win_len 7 \
    --ras_win_max_num_repeat 2
```

#### Multi-speaker TTS

For dialogue with speaker tags:

```bash
python tts.py \
    --voice_profile voice_profiles/belinda_profile.npy \
    --text "[SPEAKER0] Hello there. [SPEAKER1] Hi, how are you?" \
    --chunk_method speaker \
    --chunk_max_num_turns 2 \
    --output_audio dialogue_output.wav
```

#### Programmatic Usage

```python
from tts import VoiceProfileTTSGenerator

# Initialize generator
generator = VoiceProfileTTSGenerator(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
    device="cuda"
)

# Single-shot generation
response = generator.generate_tts_with_voice_profile_file(
    text="Hello world",
    voice_profile_path="voice_profile.npy",
    temperature=0.3
)

# Long-form generation
response = generator.generate_tts_with_voice_profile_file(
    text="Very long text...",
    voice_profile_path="voice_profile.npy",
    chunk_method="word",
    chunk_max_word_num=200,
    generation_chunk_buffer_size=5
)

# Save audio
generator.save_audio_response(response, "output.wav")
```

## Long-form TTS Features

### Chunking Methods

1. **Word-based Chunking** (`--chunk_method word`):
   - Splits text by word count
   - Respects sentence boundaries
   - Good for general long-form content
   - Configurable with `--chunk_max_word_num`

2. **Speaker-based Chunking** (`--chunk_method speaker`):
   - Splits by speaker turns
   - Ideal for dialogue and conversations
   - Configurable with `--chunk_max_num_turns`

### Context Management

- **Voice Profile Context**: Always included as reference
- **Previous Chunks**: Maintains context from recent chunks
- **Buffer Size**: Configurable with `--generation_chunk_buffer_size`

### Seamless Stitching

The system ensures smooth transitions between chunks through:
- **Prosody Continuity**: Maintains speech patterns across chunks
- **Timing Synchronization**: Preserves natural pauses and rhythm
- **Audio Quality**: Prevents artifacts at chunk boundaries

### Repetition Control

- **RAS (Repetition-Aware Sampling)**: Prevents repetitive patterns
- **Configurable Parameters**: `--ras_win_len` and `--ras_win_max_num_repeat`

## Voice Profile Format

Voice profiles are stored as `.npy` files containing:
- **Audio Tokens**: Encoded representation of voice characteristics
- **Compact Size**: Typically 1-5 MB per profile
- **Reusable**: Can be used for multiple generations

### Profile Information

```python
from vc import VoiceProfileExtractor

extractor = VoiceProfileExtractor(...)
voice_profile = extractor.load_voice_profile("profile.npy")
info = extractor.get_voice_profile_info(voice_profile)

print(f"Shape: {info['shape']}")
print(f"Size: {info['size_mb']:.2f} MB")
print(f"Token Count: {info['token_count']}")
```

## Comparison with Original Voice Cloning

| Feature | Original Voice Cloning | Voice Profile System |
|---------|----------------------|---------------------|
| **Reference Audio** | Required for each generation | Extracted once, reused |
| **Processing Time** | Slower (re-encode each time) | Faster (pre-encoded) |
| **Memory Usage** | Higher (full audio) | Lower (compressed tokens) |
| **Consistency** | May vary | Guaranteed consistency |
| **Long-form Support** | Limited | Full support with chunking |
| **Reusability** | None | High (share profiles) |

## Advanced Usage

### Batch Processing

```python
from vc import VoiceProfileExtractor
from tts import VoiceProfileTTSGenerator

# Extract multiple profiles
extractor = VoiceProfileExtractor(...)
profiles = extractor.batch_extract_profiles([
    "voice1.wav", "voice2.wav", "voice3.wav"
])

# Generate TTS for multiple texts
generator = VoiceProfileTTSGenerator(...)
responses = generator.batch_generate_tts(
    texts=["Text 1", "Text 2", "Text 3"],
    voice_profile_path="profile.npy"
)
```

### Custom Chunking

```python
from tts import prepare_chunk_text

# Custom chunking
chunks = prepare_chunk_text(
    text="Your long text here",
    chunk_method="word",
    chunk_max_word_num=150
)

print(f"Split into {len(chunks)} chunks")
```

### Error Handling

```python
try:
    response = generator.generate_tts_with_voice_profile_file(
        text="Test text",
        voice_profile_path="profile.npy"
    )
except FileNotFoundError:
    print("Voice profile not found")
except Exception as e:
    print(f"Generation failed: {e}")
```

## Troubleshooting

### Common Issues

1. **Voice Profile Not Found**:
   ```bash
   # Ensure the profile exists
   ls -la voice_profiles/
   # Re-extract if needed
   python vc.py --audio_file input.wav --output_profile voice_profiles/profile.npy
   ```

2. **Out of Memory**:
   ```bash
   # Reduce chunk size
   python tts.py --chunk_max_word_num 100 --generation_chunk_buffer_size 2
   ```

3. **Poor Audio Quality**:
   ```bash
   # Adjust generation parameters
   python tts.py --temperature 0.2 --max_new_tokens 2048
   ```

4. **Repetitive Output**:
   ```bash
   # Enable RAS
   python tts.py --ras_win_len 7 --ras_win_max_num_repeat 2
   ```

### Performance Tips

- **GPU Usage**: Use `--device cuda` for faster generation
- **Chunk Size**: Balance between memory usage and context preservation
- **Buffer Size**: Larger buffers provide better context but use more memory
- **Temperature**: Lower values (0.1-0.3) for more consistent output

## File Structure

```
higgs-audio/
├── vc.py                          # Voice profile extraction
├── tts.py                         # TTS generation (single-shot + long-form)
├── voice_profile_example.py       # Complete workflow example
├── long_form_tts_example.py       # Long-form TTS demonstration
├── test_voice_profile_system.py   # System testing
├── VOICE_PROFILE_README.md        # This documentation
├── voice_profiles/                # Extracted voice profiles
│   ├── belinda_profile.npy
│   └── chadwick_profile.npy
└── outputs/                       # Generated audio files
    ├── single_shot_tts.wav
    ├── long_form_word_chunking.wav
    └── long_form_speaker_chunking.wav
```

## Examples

### Basic Workflow

```bash
# 1. Extract voice profile
python vc.py --audio_file examples/voice_prompts/belinda.wav --output_profile voice_profiles/belinda.npy

# 2. Generate single-shot TTS
python tts.py --voice_profile voice_profiles/belinda.npy --text "Hello world" --output_audio hello.wav

# 3. Generate long-form TTS
python tts.py --voice_profile voice_profiles/belinda.npy --text "Long content..." --chunk_method word --output_audio long.wav
```

### Advanced Workflow

```bash
# Extract multiple profiles
python vc.py --audio_file voice1.wav --output_profile profiles/voice1.npy
python vc.py --audio_file voice2.wav --output_profile profiles/voice2.npy

# Generate dialogue
python tts.py \
    --voice_profile profiles/voice1.npy \
    --text "[SPEAKER0] Hello! [SPEAKER1] Hi there!" \
    --chunk_method speaker \
    --output_audio dialogue.wav
```

## Contributing

To extend the voice profile system:

1. **Add New Chunking Methods**: Modify `prepare_chunk_text()` in `tts.py`
2. **Enhance Context Management**: Extend the context buffering logic
3. **Improve Audio Quality**: Adjust generation parameters and RAS settings
4. **Add New Features**: Extend the `VoiceProfileTTSGenerator` class

## License

This system is part of the Higgs Audio V2 project and follows the same license terms. 