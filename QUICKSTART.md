# bamfscribe - Quick Start Guide

## First Time Setup

```bash
# 1. Install FFmpeg (strongly recommended for parakeet-mlx)
conda install -c conda-forge ffmpeg  # if using conda
# OR: brew install ffmpeg  # if using Homebrew

# 2. Install Python dependencies
cd ~/repos/tools/bamfscribe
./setup.sh

# 3. Set up Hugging Face token (required for speaker diarization)
# First, accept terms at these URLs:
# - https://huggingface.co/pyannote/speaker-diarization-community-1 (primary model)
# - https://huggingface.co/pyannote/segmentation-3.0
# Then get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN='your_token_here'
echo "export HF_TOKEN='your_token_here'" >> ~/.zshrc

# 4. Grant Full Disk Access (if prompted)
# System Settings → Privacy & Security → Full Disk Access → Add Terminal

# 5. Install Cursor CLI (optional, for automatic summaries)
curl https://cursor.com/install -fsS | bash
# Get API key from: https://cursor.com/dashboard?tab=cloud-agents
export CURSOR_API_KEY='your_api_key_here'
echo "export CURSOR_API_KEY='your_api_key_here'" >> ~/.zshrc
```

## Daily Usage

**Interactive selection (default)** - shows list of memos from past 7 days (✓ marks already transcribed):
```bash
cd ~/repos/tools/bamfscribe
python bamfscribe.py

# Or show recordings from past 30 days
python bamfscribe.py --ndays 30
```

After selecting a memo, you'll be prompted for speaker count (optional but speeds up diarization):
```
How many speakers? (press Enter to skip): 2
```

**Quick mode** - auto-selects latest memo (skips all prompts):
```bash
python bamfscribe.py --latest
```

**Automatic mode** - for cron jobs (processes oldest unprocessed memo from past 7 days):
```bash
python bamfscribe.py --auto

# Or search past 14 days
python bamfscribe.py --auto --ndays 14
```

**Other options:**
```bash
# Force re-process latest even if transcript exists
python bamfscribe.py --latest --force

# Maximum accuracy with large model
python bamfscribe.py --backend mlx-whisper --model large-v3 --latest

# Quick transcription with small model
python bamfscribe.py --model small --latest

# Skip automatic summary generation
python bamfscribe.py --no-summary --latest

# Process ANY audio file (works with Voice Memos or any recording)
python bamfscribe.py --file ~/path/to/recording.m4a
# Supports: .m4a, .wav, .mp3, .mp4, .caf, and other formats
```

## Choosing Backend & Model

All models are English-optimized for faster, more accurate transcription.

**Parakeet-MLX** (default): ⚡️ Exceptionally fast, English-specific
- `--backend parakeet --model medium --latest` - **Recommended** (~20x realtime)
- Only one model available (0.6b-v3), model size flag ignored
- Example: 18 minute recording transcribes in < 1 minute

**MLX-Whisper**: Uses English-specific `.en` models (tiny-medium)
- `--backend mlx-whisper --model small --latest` - Fast, English-only (~10x realtime)
- `--backend mlx-whisper --model medium --latest` - Balanced, English-only (~7x realtime)
- `--backend mlx-whisper --model large-v3 --latest` - Best accuracy, multilingual (~4x realtime)
- `--backend mlx-whisper --model large-v3-turbo --latest` - Faster large model

## Speaker Recognition

The tool automatically learns speaker voices:

**First time you transcribe:**
- System detects speakers and prompts for names
- Shows sample quotes from each speaker
- Press **m** to see more quotes (up to 5 per speaker)
- Press **p** to hear audio clips (up to 3 per speaker - press multiple times for different samples)
- Voice profiles saved to `~/.speaker_profiles.json`

**Future recordings:**
- Known speakers automatically identified
- You can confirm or correct identifications
- Audio playback and quotes still available

**Manage speakers:**
```bash
python speaker_database.py list      # View all known speakers
python speaker_database.py remove NAME  # Remove a speaker
```

See [SPEAKER_DATABASE.md](./SPEAKER_DATABASE.md) for full details.

## Output Location

Transcripts are saved to `~/transcripts/` by default:
- `.srt` file: Text format with speaker names and timestamps
- `.json` file: Structured data for further processing

## Cursor Integration

After transcription completes, the tool sends a simple prompt to Cursor:
```
Please create a summary of the transcription at: /path/to/transcript.srt
```

> ⚠️ **Privacy**: Using `cursor-agent` sends your transcript to Cursor's cloud servers for AI processing. Enable Privacy Mode in Cursor settings or use `--no-summary` to keep everything local.

**Customize with cursor rules:**
Create `.cursor/rules/*.mdc` files in your workspace to define how summaries are generated:
- Meeting summaries → Extract attendees, action items, decisions
- Personal notes → Create bullet points, highlight insights
- Interviews → Identify themes, extract quotes

See the [Integration with Cursor](./README.md#integration-with-cursor) section in README for detailed examples and configuration.

## Troubleshooting

**"Operation not permitted"**: Grant Full Disk Access in System Settings

**"No Hugging Face token"**: Transcription works, but speakers won't be labeled

**Slow first run**: Models are downloading (~2GB), subsequent runs are much faster
