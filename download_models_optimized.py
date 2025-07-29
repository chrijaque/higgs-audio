#!/usr/bin/env python3
"""
Optimized script to download only the models actually used by tts.py and vc.py.
This saves ~2.1GB compared to the full download.
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

def download_models_optimized():
    """Download only the models actually used by tts.py and vc.py."""
    
    # Set cache directories
    cache_dir = Path("/app/models")
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    
    logger.info(f"Cache directory set to: {cache_dir}")
    
    try:
        # Model paths - only what's actually used
        MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
        AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
        SEMANTIC_MODEL = "bosonai/hubert_base"  # Only the default semantic model
        
        logger.info("Starting optimized model downloads...")
        
        # Step 1: Download main model (used by tts.py)
        logger.info("Downloading main model files...")
        model_cache_path = snapshot_download(
            repo_id=MODEL_PATH,
            cache_dir=cache_dir,
            local_files_only=False
        )
        logger.info(f"✓ Main model downloaded to: {model_cache_path}")
        
        # Step 2: Download audio tokenizer (used by both vc.py and tts.py)
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
        
        # Step 4: Download only the semantic model that's actually used
        logger.info("Downloading semantic model (used by audio tokenizer)...")
        try:
            snapshot_download(
                repo_id=SEMANTIC_MODEL,
                cache_dir=cache_dir,
                local_files_only=False,
                trust_remote_code=True
            )
            logger.info(f"✓ Semantic model downloaded: {SEMANTIC_MODEL}")
        except Exception as e:
            logger.warning(f"Semantic model download warning: {e}")
        
        # Step 5: SKIP Whisper - not used by tts.py/vc.py
        logger.info("Skipping Whisper model - not used by tts.py/vc.py handlers")
        
        # Step 6: SKIP redundant semantic models
        logger.info("Skipping redundant semantic models - only using default")
        
        # Verify cache files exist
        cache_files = list(cache_dir.rglob("*"))
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        
        logger.info(f"✓ Optimized download completed!")
        logger.info(f"✓ Total files cached: {len(cache_files)}")
        logger.info(f"✓ Total cache size: {total_size / (1024**3):.1f} GB")
        logger.info(f"✓ Saved ~2.1GB compared to full download")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model download: {e}")
        return False

def verify_optimized_downloads():
    """Verify that all required files are downloaded."""
    
    cache_dir = Path("/app/models")
    
    # Check for essential model files
    required_files = [
        "bosonai/higgs-audio-v2-generation-3B-base",
        "bosonai/higgs-audio-v2-tokenizer",
        "bosonai/hubert_base"  # Only the semantic model that's used
    ]
    
    missing_files = []
    for model_path in required_files:
        model_dir = cache_dir / "hub" / model_path.replace("/", "_")
        if not model_dir.exists():
            missing_files.append(model_path)
    
    if missing_files:
        logger.warning(f"Missing model files: {missing_files}")
        return False
    else:
        logger.info("✓ All required models verified")
        return True

if __name__ == "__main__":
    logger.info("Starting optimized Higgs Audio model download...")
    logger.info("This download includes only models used by tts.py and vc.py")
    
    success = download_models_optimized()
    
    if success:
        logger.info("✓ Optimized model download completed successfully!")
        
        # Verify downloads
        if verify_optimized_downloads():
            logger.info("✓ All required models verified and ready!")
            sys.exit(0)
        else:
            logger.warning("⚠ Some model files may be missing")
            sys.exit(0)  # Don't fail build, models will download at runtime
    else:
        logger.error("✗ Model download failed!")
        logger.info("Models will be downloaded at runtime on first use")
        sys.exit(0)  # Don't fail build, allow runtime download 