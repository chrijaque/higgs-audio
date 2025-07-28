"""
Utility functions for Higgs Audio voice profile management.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from .core import VoiceProfile


def load_voice_profile(profile_path: str) -> VoiceProfile:
    """
    Load a voice profile from a .npy file.
    
    Args:
        profile_path: Path to the voice profile file
        
    Returns:
        VoiceProfile object
    """
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Voice profile file not found: {profile_path}")
    
    profile_data = np.load(profile_path, allow_pickle=True).item()
    return VoiceProfile(profile_data["tokens"], profile_data["metadata"])


def save_voice_profile(voice_profile: VoiceProfile, output_path: str):
    """
    Save a voice profile to a .npy file.
    
    Args:
        voice_profile: VoiceProfile object to save
        output_path: Path where to save the profile
    """
    profile_data = {
        "tokens": voice_profile.tokens,
        "metadata": voice_profile.metadata
    }
    np.save(output_path, profile_data)


def get_voice_profile_info(voice_profile: VoiceProfile) -> Dict[str, Any]:
    """
    Get information about a voice profile.
    
    Args:
        voice_profile: VoiceProfile object
        
    Returns:
        Dictionary with profile information
    """
    info = {
        "shape": voice_profile.shape,
        "size": voice_profile.size,
        "metadata": voice_profile.metadata
    }
    return info


def validate_voice_profile(voice_profile: VoiceProfile) -> bool:
    """
    Validate that a voice profile has the correct structure.
    
    Args:
        voice_profile: VoiceProfile object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check that tokens exist and are numpy array
        if not isinstance(voice_profile.tokens, np.ndarray):
            return False
        
        # Check that tokens have reasonable shape
        if len(voice_profile.tokens.shape) != 1:
            return False
        
        # Check that metadata exists
        if not isinstance(voice_profile.metadata, dict):
            return False
        
        return True
    except Exception:
        return False


def merge_voice_profiles(profiles: list[VoiceProfile]) -> VoiceProfile:
    """
    Merge multiple voice profiles into one.
    
    Args:
        profiles: List of VoiceProfile objects
        
    Returns:
        Merged VoiceProfile object
    """
    if not profiles:
        raise ValueError("No profiles provided")
    
    # Concatenate tokens
    all_tokens = np.concatenate([p.tokens for p in profiles])
    
    # Merge metadata
    merged_metadata = {}
    for i, profile in enumerate(profiles):
        for key, value in profile.metadata.items():
            merged_metadata[f"{key}_{i}"] = value
    
    return VoiceProfile(all_tokens, merged_metadata) 