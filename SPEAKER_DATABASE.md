# Speaker Recognition Database

Automatically identify speakers across multiple recordings using voice embeddings.

## How It Works

1. **First Recording**: System detects speakers as SPEAKER_00, SPEAKER_01, etc.
2. **Name Assignment**: You provide names for each speaker
3. **Profile Storage**: Voice embeddings saved to `~/.speaker_profiles.json`
4. **Auto-Recognition**: Future recordings automatically identify known speakers

## Usage

### Transcribe with Speaker Recognition (Default)

```bash
python bamfscribe.py
```

During/after diarization, you'll be prompted:

```
Identifying 2 speaker(s)...
SPEAKER_00 → Unknown (best match: Steven, confidence: 65%)
SPEAKER_01 → Unknown (best match: Nina, confidence: 45%)

====================================
Speaker Identification
====================================

SPEAKER_00: Unknown speaker detected
  Enter name (or press Enter to skip): Steven

SPEAKER_01: Unknown speaker detected  
  Enter name (or press Enter to skip): Nina
```

### Subsequent Recordings

On future recordings with the same people:

```
Identifying 2 speaker(s)...
SPEAKER_00 → Steven (confidence: 92%)
SPEAKER_01 → Nina (confidence: 89%)

SPEAKER_00: "Good morning everyone, let's start the meeting."
  Identified as 'Steven' (92% confidence)
  Confirm (y), rename (n), skip (s), more quotes (m), play audio (p)? [y/n/s/m/p]: p
  
    [Playing audio clip (3.2s)...]
  
  Confirm (y), rename (n), skip (s), more quotes (m), play audio (p)? [y/n/s/m/p]: y
```

## Managing the Database

### List Known Speakers

```bash
python speaker_database.py list
```

Output:
```
Known Speakers:
------------------------------------------------------------
  Steven               - seen  5x, last: 20260118_094941
  Nina                 - seen  3x, last: 20260115_143022
```

### Remove a Speaker

```bash
python speaker_database.py remove "Nina"
```

### Export Embeddings

```bash
python speaker_database.py export
# Saves to ~/.speaker_profiles.npz for analysis
```

### Disable Speaker Recognition

```bash
python bamfscribe.py --no-speaker-db
```

## How Speaker Recognition Works

**Voice Embeddings**: The diarization model creates a unique vector representation (embedding) of each speaker's voice characteristics.

**Similarity Matching**: New embeddings are compared to known speakers using cosine similarity (0-1 scale).

**Threshold**: Default 75% similarity required for automatic match. Lower matches prompt for confirmation.

**Profile Updates**: Each time a speaker is confirmed, their profile is updated with a weighted average of embeddings, allowing adaptation to voice changes over time.

## Interactive Features

During speaker identification, you have several options to help identify speakers:

**Sample Quotes**: See what each speaker said to help identify them
- Press **m** to see more quotes (up to 5 per speaker)

**Audio Playback**: Hear each speaker's voice
- Press **p** to play a 2-8 second audio clip
- Up to 3 different clips per speaker - press **p** multiple times to hear different samples
- Uses built-in macOS `afplay` command
- Helps distinguish speakers with similar speech patterns

**Example:**
```
SPEAKER_00: "Good morning everyone, let's start the meeting."
  Unknown speaker detected
  Enter name, more quotes (m), play audio (p), or press Enter to skip: m

  Additional quotes from SPEAKER_00:
    2. "I think we should focus on the timeline first."
    3. "That's a great point, let me write that down."
  
  Enter name, more quotes (m), play audio (p), or press Enter to skip: p
  
    [Playing audio clip 1/3, 3.2s...]
  
  Enter name, more quotes (m), play audio (p), or press Enter to skip: p
  
    [Playing audio clip 2/3, 4.5s...]
  
  Enter name, more quotes (m), play audio (p), or press Enter to skip: Steven
```

## Database Location

- **Profiles**: `~/.speaker_profiles.json`
- **Format**: JSON with speaker names, embeddings, and metadata
- **Privacy**: Stored locally, never leaves your computer

## Tips

**For Best Results:**
- Name speakers consistently (use full names or consistent nicknames)
- Confirm identifications when confidence is below 90%
- If someone's voice isn't recognized after multiple recordings, consider removing and re-adding their profile

**Multi-Person Meetings:**
- After each meeting, take a moment to name any new participants
- Known participants will be automatically labeled in future meetings
- This makes meeting summaries much more useful

**Voice Note Recognition (Single Speaker):**
- If recording only yourself, the tool will learn your voice profile
- Future solo recordings will automatically label you
- Useful for distinguishing your notes from meetings

## Troubleshooting

**Speaker misidentified:**
- Choose "rename" (n) when prompted
- Enter correct name
- Their profile will be updated

**Confidence too low:**
- Background noise, poor audio quality, or voice changes can affect recognition
- Consider re-recording in better conditions
- System adapts over time as it sees more examples

**Database corruption:**
- Delete `~/.speaker_profiles.json`  
- Rebuild by transcribing a few recordings and naming speakers again

## Privacy & Security

- All voice profiles stored locally
- No cloud sync or external storage
- Embeddings are mathematical representations, not actual audio
- Safe to backup/sync as regular JSON file
