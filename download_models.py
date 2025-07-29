#!/usr/bin/env python3
"""
Script to download Higgs Audio models during Docker build.
This script actually triggers the model downloads and caches them.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required models and cache them."""
    
    # Set cache directories
    cache_dir = Path("/app/models")
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    
    logger.info(f"Cache directory set to: {cache_dir}")
    
    try:
        # Model paths
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        
        logger.info("Starting model downloads...")
        
        # Step 1: Download the main model files
        logger.info("Downloading main model files...")
        model_cache_path = snapshot_download(
            repo_id=MODEL_PATH,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logger.info(f"✓ Main model downloaded to: {model_cache_path}")
        
        # Step 2: Download the audio tokenizer files
        logger.info("Downloading audio tokenizer files...")
        tokenizer_cache_path = snapshot_download(
            repo_id=AUDIO_TOKENIZER_PATH,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logger.info(f"✓ Audio tokenizer downloaded to: {tokenizer_cache_path}")
        
        # Step 3: Download tokenizer config (needed for AutoTokenizer)
        logger.info("Downloading tokenizer configuration...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=cache_dir)
            config = AutoConfig.from_pretrained(MODEL_PATH, cache_dir=cache_dir)
            logger.info("✓ Tokenizer and config downloaded")
        except Exception as e:
            logger.warning(f"Tokenizer download warning: {e}")
        
        # Step 4: Download additional dependencies
        logger.info("Downloading additional dependencies...")
        
        # Download semantic models used by the audio tokenizer
        semantic_models = [
            "facebook/hubert-base-ls960",
            "microsoft/wavlm-base-plus",
            "bosonai/hubert_base"
        ]
        
        for model_id in semantic_models:
            try:
                logger.info(f"Downloading {model_id}...")
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                logger.info(f"✓ {model_id} downloaded")
            except Exception as e:
                logger.warning(f"Failed to download {model_id}: {e}")
        
        # Step 5: Download Whisper model (if needed)
        try:
            logger.info("Downloading Whisper model...")
            from transformers import AutoProcessor
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                cache_dir=cache_dir,
                trust_remote=True
            )
            logger.info("✓ Whisper processor downloaded")
        except Exception as e:
            logger.warning(f"Whisper download warning: {e}")
        
        # Verify cache files exist
        cache_files = list(cache_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        
        logger.info(f"✓ Download completed!")
        logger.info(f"✓ Total files cached: {len(cache_files)}")
        logger.info(f"✓ Total cache size: {total_size / (1024**3):.1f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model download: {e}")
        return False

def verify_downloads():
    """Verify that all required files are downloaded."""
    
    cache_dir = Path("/app/models")
    
    # Check for main model files
    model_files = [
        "bosonai/higgs-audio-v2-generation-3B-base",
        "bosonai/higgs-audio-v2-tokenizer"
    ]
    
    missing_files = []
    for model_path in model_files:
        model_dir = cache_dir / "hub" / model_path.replace("/", "_")
        if not model_dir.exists():
            missing_files.append(model_path)
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
        return False
    else:
        logger.info("✓ All model files verified")
        return True

if __name__ == "__main__":
    logger.info("Starting Higgs Audio model download...")
    
    success = download_models()
    
    if success:
        logger.info("✓ Model download completed successfully!")
        
        # Verify downloads
        if verify_downloads():
            logger.info("✓ All models verified and ready!")
            sys.exit(0)
        else:
            logger.warning("⚠ Some model files may be missing")
            sys.exit(0)  # Don't fail build, models will download at runtime
    else:
        logger.error("✗ Model download failed!")
        logger.info("Models will be downloaded at runtime on first use")
        sys.exit(0)  # Don't fail build, allow runtime download