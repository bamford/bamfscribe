# bamfscribe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](https://support.apple.com/en-us/116943)

**Local transcription and diarization of Apple Voice Memos using open-source models**

Automatically transcribe voice memos and meeting recordings with speaker identification, then optionally generate structured summaries using Cursor AI.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Integration with Cursor](#integration-with-cursor)
- [Speaker Recognition](#speaker-recognition)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- 🎙️ **Flexible input** - process Voice Memos, or any audio file (m4a, wav, mp3, etc.)
- 📋 **Interactive selection** - choose from last 10 recordings or auto-select latest
- 📝 **Local transcription** using English-optimized models (parakeet-mlx or mlx-whisper)
- 👥 **Speaker diarization** using pyannote.audio (community-1 model - 2.5% real-time factor)
- 🧠 **Speaker recognition** - automatically identifies known speakers across recordings
- 🔊 **Audio playback** - hear multiple voice clips during speaker identification (up to 3 samples per speaker)
- 💬 **Sample quotes** - see what each speaker said to help identify them
- 💾 **SRT output** with speaker names (not just SPEAKER_00, SPEAKER_01)
- 📊 **Progress tracking** - chunk progress for transcription, real-time progress bars for diarization, timing summary for all steps
- 🤖 **Cursor integration** for automatic summary generation (optional, requires cloud processing)
- ⚡️ **Optimized for English** - faster and more accurate
- 🔒 **Local processing** - transcription and speaker identification run entirely on your Mac

## Technology Stack

- **Transcription**: 
  - **parakeet-mlx** (recommended, faster processing, comparable accuracy)
  - **mlx-whisper** (OpenAI Whisper optimized for Apple Silicon)
- **Diarization**: pyannote.audio speaker-diarization-community-1 (2.5% real-time factor)
- **Speaker Recognition**: Custom embedding-based database with cosine similarity
- **Summary Generation**: Cursor AI CLI
- **Language**: Python 3.12+

### Backend Comparison

| Feature | Parakeet-MLX | MLX-Whisper |
|---------|--------------|-------------|
| **Speed** | ⚡️ ~20x realtime (exceptionally fast) | ~4-10x realtime |
| **Accuracy** | Excellent for English | Excellent |
| **Best for** | **All meetings, voice notes** | When multi-language needed |
| **Models** | 0.6b-v3 (English-specific) | tiny → large-v3-turbo (English `.en` available) |
| **Memory** | Lower usage (~2.5GB model) | Medium to high usage |
| **Default** | ✅ Yes | Available |
| **English-only** | ✅ Optimized | ✅ `.en` models for tiny-medium |
| **Real-world** | 18 min audio → <1 min | 18 min audio → ~2-3 min |

**Note**: Parakeet-MLX is dramatically faster (~20x realtime) while maintaining excellent accuracy for English. For most use cases, there's no reason to use anything else.

**Recommendation**: Use parakeet (default). Only switch to mlx-whisper if you need multilingual support or want to experiment with different model sizes.

## Installation

### Prerequisites

1. Python 3.12 or higher
2. Apple Silicon Mac (for optimal performance)
3. **FFmpeg** (strongly recommended - required for parakeet-mlx and accurate audio file handling)
4. Hugging Face account (for speaker diarization)
5. Cursor CLI and API key (optional, for automatic summary generation)

### Setup

1. **Install FFmpeg** (strongly recommended):

```bash
# Using conda (if in a conda environment):
conda install -c conda-forge ffmpeg

# OR using Homebrew (system-wide):
brew install ffmpeg
```

**Important**: FFmpeg is strongly advised for parakeet-mlx (the default transcription backend) and ensures accurate audio file handling. Without it:
- Parakeet-mlx may have issues loading audio files
- Duration detection will be inaccurate (showing wrong times in the selection menu)
- Some audio formats may not work properly

2. **Install Python dependencies:**

```bash
cd ~/repos/tools/bamfscribe
pip install -r requirements.txt

# Note: Includes scikit-learn for speaker recognition
```

3. **Set up Hugging Face token for speaker diarization:**

   a. Create account at [https://huggingface.co](https://huggingface.co)
   
   b. Accept terms for required models:
      - **[https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)** (primary - fastest)
      - [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   
   c. Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   
   d. Set environment variable:
   ```bash
   export HF_TOKEN='your_token_here'
   # Add to ~/.zshrc to make permanent:
   echo "export HF_TOKEN='your_token_here'" >> ~/.zshrc
   ```

4. **Grant Full Disk Access** (required to read Voice Memos):

   - System Settings → Privacy & Security → Full Disk Access
   - Add Terminal or your Python interpreter

5. **Install Cursor CLI** (optional, for automatic summary generation):

   a. Install using the official installer:
   ```bash
   curl https://cursor.com/install -fsS | bash
   ```
   
   Installation instructions: [https://cursor.com/docs/cli/installation](https://cursor.com/docs/cli/installation)
   
   b. Get your Cursor API key from [https://cursor.com/dashboard?tab=cloud-agents](https://cursor.com/dashboard?tab=cloud-agents)
   
   c. Set environment variable:
   ```bash
   export CURSOR_API_KEY='your_api_key_here'
   # Add to ~/.zshrc to make permanent:
   echo "export CURSOR_API_KEY='your_api_key_here'" >> ~/.zshrc
   ```
   
   d. Verify installation:
   ```bash
   agent --version
   # or
   cursor-agent --version
   ```

## Usage

### Basic Usage

**Interactive selection (default):**

```bash
python bamfscribe.py
```

Shows a list of your last 10 voice memos with dates, times, and durations. You select which one to process.

After selection, you'll be prompted for the number of speakers (optional, but significantly speeds up diarization):
```
How many speakers? (press Enter to skip): 2
```

**Auto-select latest:**

```bash
python bamfscribe.py --latest
```

Automatically processes the most recent memo without prompting (skips both file selection and speaker count).

**Process a specific audio file:**

```bash
python bamfscribe.py --file /path/to/recording.m4a
```

Process any audio file (not just Voice Memos). Supports `.m4a`, `.mp4`, `.m4v`, `.caf`, `.wav`, and other common audio formats. You'll be prompted for the number of speakers after selecting the file.

**Processing flow:**
1. Choose input:
   - Interactive: List and select from last 10 Voice Memos
   - Auto-select: `--latest` for most recent Voice Memo
   - Direct file: `--file path/to/audio.m4a` for any audio file
2. Optionally specify number of speakers (speeds up diarization)
3. Transcribe with parakeet-mlx
   - Shows chunk progress for long recordings (>20 min)
   - Timing information displayed
4. Identify speakers with pyannote
   - Real-time progress bars
   - Timing information displayed
5. **Match speakers to known profiles** with:
   - Sample quotes from each speaker (up to 5 per speaker)
   - Audio playback - press 'p' to hear their voice (up to 3 clips per speaker)
   - Press 'm' for more quotes, press 'p' multiple times for different audio samples
6. Save SRT and JSON transcripts with speaker names
7. Generate meeting summary with Cursor (optional)
8. Display timing summary for all steps

See [SPEAKER_DATABASE.md](./SPEAKER_DATABASE.md) for details on speaker recognition features.

### Choosing Transcription Backend

**Parakeet-MLX (recommended)**: Faster processing with comparable accuracy
```bash
python bamfscribe.py --backend parakeet --model medium --latest
```

**MLX-Whisper**: OpenAI Whisper model, slightly more accurate for some use cases
```bash
python bamfscribe.py --backend mlx-whisper --model large-v3 --latest
```

### Model Sizes

Both backends support multiple model sizes (speed vs accuracy trade-off):

**Parakeet models** (English-specific):
- `tiny` - Fastest, least accurate (~60MB)
- `small` - Good balance for quick notes (~250MB)
- `medium` - **Recommended default**, excellent accuracy/speed (~500MB)
- Large models not yet available

**MLX-Whisper models** (English-optimized `.en` variants):
- `tiny`, `base`, `small` - Fast but less accurate (~40-250MB)
- `medium` - Good balance (~800MB)
- `large` or `large-v3` - **Best accuracy**, slower (~1.5GB, multilingual)
- `large-v3-turbo` - Faster large model with similar accuracy

Note: For English audio, the `.en` models (tiny-medium) are 20-30% faster than multilingual versions.

Examples:
```bash
# Quick transcription for personal notes (auto-select latest)
python bamfscribe.py --backend parakeet --model small --latest

# Maximum accuracy for important meetings (interactive selection)
python bamfscribe.py --backend mlx-whisper --model large-v3

# Balanced approach (default, interactive)
python bamfscribe.py --backend parakeet --model medium
```

### Advanced Options

```bash
# Auto-select latest without prompt
python bamfscribe.py --latest

# Process ANY audio file (not just Voice Memos)
python bamfscribe.py --file /path/to/recording.m4a
# Supports: .m4a, .mp4, .m4v, .caf, .wav, and other common formats

# Specify custom voice memos directory
python bamfscribe.py --voice-memos-dir /path/to/recordings

# Specify custom output directory
python bamfscribe.py --output-dir /path/to/transcripts --latest

# Skip summary generation
python bamfscribe.py --no-summary --latest


# Disable speaker recognition
python bamfscribe.py --no-speaker-db --latest

# Speed up diarization by specifying speaker count (recommended!)
python bamfscribe.py --num-speakers 2 --latest

# Combine options
python bamfscribe.py --backend mlx-whisper --model large-v3 --num-speakers 2
```

### Setting Defaults (Optional)

Create a `config.py` file to set your preferred defaults:

```bash
cp config.example.py config.py
# Edit config.py with your preferences
```

Key configuration options:
- `DEFAULT_BACKEND`: "parakeet" or "mlx-whisper"
- `DEFAULT_MODEL_SIZE`: Model size to use
- `OUTPUT_DIR`: Where to save transcripts (default: `~/transcripts`)
- `CURSOR_WORKSPACE_PATH`: Absolute path to your workspace containing `.cursor/rules/*.mdc` files for customizing summary generation
- `GENERATE_SUMMARY`: Whether to generate summaries by default

This avoids having to specify the same options every time.

## Output

The tool generates two files in the output directory (default: `~/transcripts/`):

1. **SRT file** (`recording_name_timestamp.srt`):
   - Standard subtitle format
   - Speaker labels: `[SPEAKER_00]`, `[SPEAKER_01]`, etc.
   - Timestamps for each segment
   - Compatible with video players and text editors

2. **JSON file** (`recording_name_timestamp.json`):
   - Machine-readable format
   - Detailed segment information
   - Useful for further processing

### Example SRT Output (With Speaker Recognition)

```
1
00:00:00,000 --> 00:00:04,250
[Steven] Good morning everyone, let's start the meeting.

2
00:00:04,500 --> 00:00:08,750
[Nina] Thanks for organizing this. I have a few updates to share.

3
00:00:09,000 --> 00:00:12,500
[Steven] Great, please go ahead.
```

After first identifying speakers and saving their profiles, future recordings
will automatically use real names instead of SPEAKER_00, SPEAKER_01, etc.

## Integration with Cursor

After transcription, the tool can automatically invoke Cursor to generate a summary of your transcription using AI.

> ⚠️ **Privacy Note**: When using `cursor-agent`, your transcript will be sent to Cursor's servers for AI processing. While all transcription and speaker diarization happens locally on your Mac, summary generation requires cloud processing. To enhance privacy, consider enabling **Privacy Mode** in Cursor's settings, which prevents Cursor from storing your data. Alternatively, use `--no-summary` to skip automatic summary generation and process transcripts manually.

### Requirements

1. **Install Cursor CLI**: Follow instructions at [https://cursor.com/docs/cli/installation](https://cursor.com/docs/cli/installation)
2. **Set API Key**: Get your key from [https://cursor.com/dashboard?tab=cloud-agents](https://cursor.com/dashboard?tab=cloud-agents) and set `CURSOR_API_KEY` environment variable
3. **Configure Workspace** (optional): Set `CURSOR_WORKSPACE_PATH` in `config.py` to your workspace folder
4. **Enable Privacy Mode** (recommended): In Cursor Settings → Privacy, enable Privacy Mode to prevent data storage on Cursor's servers

### How It Works

The tool sends a simple prompt to Cursor:
```
Please create a summary of the transcription at: /path/to/transcript.srt
```

Cursor then:
- Uses `cursor-agent` (or `agent`) with `--force`, `--stream-partial-output`, and `--workspace` flags
- Reads the transcript file
- Generates a summary based on your cursor rules (if configured)
- Can save the summary to your notes folder (if rules specify)

### Customizing with Cursor Rules

You can customize how Cursor processes transcriptions by creating `.cursor/rules/*.mdc` files in your workspace. This allows you to:

**Define different processing for different contexts:**
```markdown
<!-- .cursor/rules/meeting-summaries.mdc -->
When summarizing meeting transcriptions:
- Include meeting title, date, and attendees
- Extract action items
- Summarize key discussion points
- Save to daily notes folder
- Link action items to task list
```

```markdown
<!-- .cursor/rules/personal-notes.mdc -->
When summarizing personal voice notes:
- Create bullet-point summary
- Highlight key insights
- Extract any tasks or reminders
- Save to inbox
```

```markdown
<!-- .cursor/rules/interviews.mdc -->
When summarizing interviews:
- Identify main themes
- Extract interesting quotes
- Note follow-up questions
- Organize by topic
```

**Provide context locations:**
You can tell Cursor where to find relevant context:
- Recent meeting notes in `~/notes/meetings/`
- Project documentation in `~/notes/projects/`
- Task list at `~/notes/tasks.md`

**Example rule file:**
```markdown
<!-- .cursor/rules/transcription-processing.mdc -->
# Transcription Processing

When processing transcriptions from bamfscribe:

1. Read the .srt file for the full transcript
2. Check the speaker names to understand who was present
3. Look in ~/notes/meetings/ for recent related notes
4. Create a structured summary with:
   - Meeting metadata (date, attendees, duration)
   - Key discussion points
   - Decisions made
   - Action items with owners
   - Follow-up questions
5. Save to ~/notes/meetings/YYYY-MM-DD-title.md
6. Update ~/notes/tasks.md with any action items

Use British English spelling and markdown formatting.
```

This flexible approach allows each user to customize summary generation for their specific workflow and note-taking system.

## Troubleshooting

### "Operation not permitted" error

Voice Memos folder requires Full Disk Access. Grant permission in System Settings → Privacy & Security.

### "No Hugging Face token" warning

Speaker diarization requires a HF token with access to pyannote models. 

1. Create account at https://huggingface.co
2. Accept terms for these models:
   - **https://huggingface.co/pyannote/speaker-diarization-community-1** (primary - 2.5% real-time factor)
   - https://huggingface.co/pyannote/segmentation-3.0
3. Get token from https://huggingface.co/settings/tokens
4. Set: `export HF_TOKEN='your_token_here'`

Without a token, the tool will transcribe but won't label speakers.

### Slow transcription

First run downloads the model (~2.5GB for parakeet, ~1.5GB for whisper). Subsequent runs are much faster. With parakeet, expect ~20x realtime speed on Apple Silicon.

### Models not downloading

Check your internet connection. The tool downloads:
- Parakeet medium model (~500MB) or Whisper large-v3 model (~1.5GB)
- Pyannote diarization model (~500MB)

### Memory issues

For very long recordings (>2 hours), you may need 16GB+ RAM. Consider splitting long recordings or using a smaller Whisper model variant.

### Privacy concerns with Cursor

**Q: I'm concerned about sending transcripts to Cursor's servers. What are my options?**

A: You have several options:
1. **Enable Privacy Mode**: In Cursor Settings → Privacy, enable Privacy Mode to prevent Cursor from storing your data on their servers
2. **Skip automatic summaries**: Use the `--no-summary` flag to keep all processing local
3. **Manual processing**: Generate transcripts locally, then manually create summaries in Cursor when needed
4. **Selective use**: Only use automatic summaries for non-sensitive content

Remember: Transcription, diarization, and speaker recognition all happen 100% locally regardless of summary settings.

### Cursor CLI not found

If you see "Warning: Cursor CLI not found":

1. Install Cursor CLI: `curl https://cursor.com/install -fsS | bash`
2. Verify installation: `agent --version` or `cursor-agent --version`
3. Make sure `~/.local/bin` is in your PATH:
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Cursor authentication errors

If you see "401" or "unauthorized" errors:

1. Get your API key from [https://cursor.com/dashboard?tab=cloud-agents](https://cursor.com/dashboard?tab=cloud-agents)
2. Set the environment variable:
   ```bash
   export CURSOR_API_KEY='your_api_key_here'
   echo "export CURSOR_API_KEY='your_api_key_here'" >> ~/.zshrc
   ```
3. Restart your terminal or run `source ~/.zshrc`

### Summary generation issues

If automatic summary generation fails, you can manually create a summary:

1. Open the transcript file (path shown in error message)
2. Use Cursor with the `meeting-summaries` rule to create a summary
3. Save it in your daily notes folder

## Performance

The tool provides progress feedback and detailed timing information:

**During processing:**
```
Transcribing with parakeet-mlx (medium)...
Recording duration: 42:15
Using chunking for memory efficiency...
Chunking enabled: 20.0 min chunks, 5.0 min overlap
  Processing chunk 1/3...
  Processing chunk 2/3...
  Processing chunk 3/3...

Diarizing audio...
speaker_diarization: 100%|████████████| 1/1 [00:38<00:00, 38.24s/it]
```

**After completion:**
```
============================================================
TIMING SUMMARY
============================================================
Audio duration: 40:00
Transcription: 52.3s
Diarization:   40.2s
Merge/Speaker: 1.2s
Save files:    0.3s
Summary:       12.5s
Total:         106.3s
Real-time factor: 0.04x
```

Typical processing times on M1/M2/M3 Mac (English-optimized models):

**With parakeet-mlx (0.6b-v3 model, English-specific)** - ⚡️ Fastest:
- 10 minute recording: ~30 seconds (~20x realtime)
- 18 minute recording: < 1 minute (~20x realtime)
- 30 minute recording: ~1.5 minutes
- 1 hour recording: ~3 minutes

**With mlx-whisper (medium.en model, English-optimized)**:
- 10 minute recording: ~1-1.5 minutes (~7-10x realtime)
- 30 minute recording: ~3-4 minutes
- 1 hour recording: ~7-10 minutes

**With mlx-whisper (large-v3 model, multilingual)**:
- 10 minute recording: ~2-3 minutes (~3-5x realtime)
- 30 minute recording: ~5-8 minutes
- 1 hour recording: ~12-18 minutes

**Speaker Diarization** (speaker-diarization-community-1):
- 2.5% real-time factor
- 40 minute recording: ~1 minute (much faster with `--num-speakers 2`)
- 1 hour recording: ~1.5 minutes

**Note**: Parakeet-mlx is exceptionally fast (~20x realtime) with excellent accuracy for English. Diarization is also very fast with the community-1 model, especially when you specify the number of speakers.

## Privacy & Security

### What Stays Local (100% Private)

- **Transcription**: Audio processing happens entirely on your Mac using local models
- **Speaker Diarization**: Speaker detection runs locally with pyannote
- **Speaker Recognition**: Voice profiles stored locally in `~/.speaker_profiles.json`
- **Models**: parakeet-mlx, mlx-whisper, and pyannote run entirely on-device
- **Audio Files**: Your voice memos never leave your machine

### What Uses Cloud Processing

- **Cursor Summary Generation** (optional feature):
  - When enabled, **transcript text** is sent to Cursor's servers for AI-powered summarization
  - Your audio files are NOT sent - only the generated text transcript
  - This feature is optional - use `--no-summary` to disable it
  - Enable Privacy Mode in Cursor settings to prevent Cursor from storing your data
  - You can always generate summaries manually later

**Bottom line**: Transcription and speaker identification are completely private and local. Only the optional summary generation feature sends transcript text to the cloud.

## File Locations

- **Voice Memos**: `~/Library/Group Containers/group.com.apple.voicememos.shared/Recordings/`
- **Transcripts**: `~/transcripts/` (configurable)
- **Models Cache**: `~/.cache/huggingface/` and `~/.cache/mlx/`

## Project Structure

```
bamfscribe/
├── bamfscribe.py           # Main script
├── speaker_database.py     # Speaker recognition module
├── config.example.py       # Configuration template
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── test_installation.py   # Installation verification
├── README.md              # This file
├── QUICKSTART.md          # Quick start guide
├── SPEAKER_DATABASE.md    # Speaker recognition docs
├── TESTING.md             # Testing documentation
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # MIT License
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Before submitting a PR:
1. Test your changes with `python test_installation.py`
2. Ensure documentation is updated
3. Follow PEP 8 style guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

This project was coded by [Steven Bamford](https://github.com/bamford) with the help of [Cursor](https://cursor.com) and Anthropic's Claude Sonnet 4.5.

**Technology Stack:**
- [parakeet-mlx](https://github.com/senstella/parakeet-mlx) - Fast English transcription
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) - Apple Silicon optimized Whisper
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Cursor](https://cursor.com) - AI-powered development environment

## Support

- 📖 [Quick Start Guide](./QUICKSTART.md)
- 🐛 [Report a Bug](https://github.com/bamford/bamfscribe/issues/new?template=bug_report.md)
- 💡 [Request a Feature](https://github.com/bamford/bamfscribe/issues/new?template=feature_request.md)
- 📧 Open an issue for questions

## Keywords

voice-memo, transcription, diarization, whisper, mlx, pyannote, speaker-identification, meeting-summary, local-processing, privacy, apple-silicon, python

---

**Made with ❤️ for Apple Silicon Macs**

*Coded by Steven Bamford with the help of Cursor and Claude Sonnet 4.5*
