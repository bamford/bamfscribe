#!/usr/bin/env python
"""
Test script to verify installation of transcription dependencies.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required = {
        "torch": "PyTorch",
        "pyannote.audio": "Pyannote Audio",
    }
    
    optional = {
        "mlx_whisper": "MLX-Whisper",
        "parakeet_mlx": "Parakeet-MLX",
    }
    
    all_ok = True
    
    # Test required packages
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED (required)")
            all_ok = False
    
    # Test optional packages (at least one needed)
    transcription_ok = False
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✓ {name}")
            transcription_ok = True
        except ImportError:
            print(f"○ {name} - not installed (optional)")
    
    if not transcription_ok:
        print("\n✗ ERROR: No transcription backend installed!")
        print("Install at least one of: mlx-whisper or parakeet-mlx")
        all_ok = False
    
    return all_ok

def test_hf_token():
    """Check if Hugging Face token is set."""
    import os
    
    print("\nTesting Hugging Face token...")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    
    if hf_token:
        print("✓ HF_TOKEN is set")
        return True
    else:
        print("○ HF_TOKEN not set (speaker diarization won't work)")
        print("  Get token from: https://huggingface.co/settings/tokens")
        print("  Accept terms: https://huggingface.co/pyannote/speaker-diarization")
        return False

def test_voice_memos_access():
    """Check if Voice Memos directory is accessible."""
    from pathlib import Path
    
    print("\nTesting Voice Memos access...")
    voice_memos_dir = Path.home() / "Library/Group Containers/group.com.apple.voicememos.shared/Recordings"
    
    if voice_memos_dir.exists():
        try:
            list(voice_memos_dir.glob("*"))
            print("✓ Voice Memos directory accessible")
            return True
        except PermissionError:
            print("✗ Voice Memos directory exists but not accessible")
            print("  Grant Full Disk Access in System Settings")
            return False
    else:
        print("○ Voice Memos directory not found")
        print(f"  Expected: {voice_memos_dir}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Voice Memo Transcriber - Installation Test")
    print("=" * 60)
    print()
    
    imports_ok = test_imports()
    hf_ok = test_hf_token()
    voice_memos_ok = test_voice_memos_access()
    
    print()
    print("=" * 60)
    
    if imports_ok and voice_memos_ok:
        print("✓ All required tests passed!")
        if not hf_ok:
            print("Note: Speaker diarization requires HF_TOKEN")
        print()
        print("Ready to use! Try:")
        print("  python bamfscribe.py --help")
        return 0
    else:
        print("✗ Some tests failed. See messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
