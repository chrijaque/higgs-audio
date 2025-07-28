"""
Command-line interface for Higgs Audio.
"""

from .vc import main as vc_main
from .tts import main as tts_main

__all__ = ["vc_main", "tts_main"] 