"""
Configuration defaults for voice memo transcription.

Copy this file to config.py and modify as needed.
"""

# Transcription backend: "parakeet" or "mlx-whisper"
DEFAULT_BACKEND = "parakeet"

# Model size
# For parakeet: "tiny", "small", "medium"
# For mlx-whisper: "tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"
DEFAULT_MODEL_SIZE = "medium"

# Voice memos directory (None = use default Apple location)
VOICE_MEMOS_DIR = None

# Output directory for transcripts (None = ~/transcripts)
OUTPUT_DIR = None

# Cursor workspace path for summary generation
# Set this to your workspace folder where cursor rules are defined
# Example: "/Users/username/notes" or "/Users/username/Documents/workspace"
# Cursor will use .cursor/rules/*.mdc files in this folder to process transcriptions
CURSOR_WORKSPACE_PATH = None

# Whether to generate summary by default
GENERATE_SUMMARY = True
