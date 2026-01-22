#!/usr/bin/env python
"""
bamfscribe - Voice Memo Transcriber with Speaker Diarization

Transcribes the latest Apple Voice Memo recording using local open-source models
(ParakeetMLX for transcription, pyannote for speaker diarization), then generates
a meeting summary using Cursor CLI.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

# Heavy imports delayed until after file selection
torch = None
Pipeline = None
ProgressHook = None
SpeakerDatabase = None
mlx_whisper = None
parakeet = None

def load_heavy_imports():
    """Load heavy dependencies only when needed."""
    global torch, Pipeline, SpeakerDatabase, ProgressHook
    
    if torch is not None:
        return  # Already loaded
    
    try:
        import torch as _torch
        from pyannote.audio import Pipeline as _Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook as _ProgressHook
        from speaker_database import SpeakerDatabase as _SpeakerDatabase
        
        torch = _torch
        Pipeline = _Pipeline
        ProgressHook = _ProgressHook
        SpeakerDatabase = _SpeakerDatabase
    except ImportError as e:
        print(f"Error: Missing required package. Please install dependencies first.")
        print(f"Run: pip install -r requirements.txt")
        print(f"Details: {e}")
        sys.exit(1)


class VoiceMemoTranscriber:
    """Handles transcription and diarization of voice memos."""
    
    # Available models for each backend (English-optimized)
    WHISPER_MODELS = {
        "tiny": "mlx-community/whisper-tiny.en-mlx",
        "base": "mlx-community/whisper-base.en-mlx", 
        "small": "mlx-community/whisper-small.en-mlx",
        "medium": "mlx-community/whisper-medium.en-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",  # No .en variant for v3
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo"
    }
    
    PARAKEET_MODELS = {
        "tiny": "mlx-community/parakeet-tdt-0.6b-v3",
        "small": "mlx-community/parakeet-tdt-0.6b-v3", 
        "medium": "mlx-community/parakeet-tdt-0.6b-v3"
    }
    
    def __init__(self, voice_memos_dir=None, output_dir=None, 
                 backend="parakeet", model_size="medium", use_speaker_db=True, num_speakers=None,
                 speaker_confidence_threshold=0.95, ndays=7):
        """
        Initialize the transcriber.
        
        Args:
            voice_memos_dir: Path to Voice Memos recordings folder
            output_dir: Path to save transcripts (defaults to ~/transcripts)
            backend: Transcription backend ("mlx-whisper" or "parakeet")
            model_size: Model size (tiny/base/small/medium/large/large-v3/large-v3-turbo)
            use_speaker_db: Enable speaker recognition database
            num_speakers: Number of speakers (if known, speeds up diarization significantly)
            speaker_confidence_threshold: Only prompt for confirmation if confidence < threshold (default: 0.95)
            ndays: Number of days back to search for recordings (default: 7)
        """
        if voice_memos_dir is None:
            voice_memos_dir = Path.home() / "Library/Group Containers/group.com.apple.voicememos.shared/Recordings"
        
        if output_dir is None:
            output_dir = Path.home() / "transcripts"
        
        self.voice_memos_dir = Path(voice_memos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend.lower()
        self.model_size = model_size.lower()
        self.use_speaker_db = use_speaker_db
        self.num_speakers = num_speakers
        self.speaker_confidence_threshold = speaker_confidence_threshold
        self.ndays = ndays
        self.speaker_db = None  # Loaded lazily when needed
        
        # Validate backend
        if self.backend not in ["mlx-whisper", "parakeet"]:
            print(f"Error: Unknown backend '{backend}'. Use 'mlx-whisper' or 'parakeet'.")
            sys.exit(1)
        
        # Check if voice memos directory exists and is accessible
        if not self.voice_memos_dir.exists():
            print(f"Error: Voice Memos directory not found: {self.voice_memos_dir}")
            print("Make sure the path is correct.")
            sys.exit(1)
    
    def get_audio_duration(self, audio_file):
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Duration in seconds, or None if unable to determine
        """
        # Try ffprobe first (most reliable for m4a/mp4 containers)
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_file)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                return duration
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass  # ffprobe not available or failed
        
        # Fallback to soundfile (works for wav, flac, etc.)
        try:
            import soundfile as sf
            with sf.SoundFile(str(audio_file)) as f:
                duration = len(f) / f.samplerate
                return duration
        except Exception:
            pass
        
        # Last resort: estimate from file size (very rough)
        # ~1MB per minute for voice memo quality
        try:
            size_mb = audio_file.stat().st_size / (1024 * 1024)
            return size_mb * 60  # rough estimate
        except Exception:
            return None
    
    def list_voice_memos(self, limit=None):
        """
        List recent voice memos with metadata.
        
        Args:
            limit: Maximum number of files to return (None = no limit, filtered by ndays)
            
        Returns:
            List of (file_path, metadata_dict) tuples, sorted newest first
        """
        from datetime import timedelta
        
        # Look for common audio formats
        audio_extensions = [".m4a", ".mp4", ".m4v", ".caf"]
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.voice_memos_dir.glob(f"*{ext}"))
        
        if not audio_files:
            return []
        
        # Get metadata for each file (including parsed date from filename)
        memos = []
        for audio_file in audio_files:
            stat = audio_file.stat()
            duration = self.get_audio_duration(audio_file)
            
            # Try to parse timestamp from filename (format: YYYYMMDD HHMMSS-ID.ext)
            name = audio_file.stem  # filename without extension
            try:
                # Extract date and time parts (first 15 chars: "YYYYMMDD HHMMSS")
                if len(name) >= 15 and name[8] == " ":
                    date_str = name[:8]  # YYYYMMDD
                    time_str = name[9:15]  # HHMMSS
                    
                    parsed_time = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
                    date_display = parsed_time.strftime("%Y-%m-%d")
                    time_display = parsed_time.strftime("%H:%M:%S")
                else:
                    # Fallback to file modification time
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    date_display = mod_time.strftime("%Y-%m-%d")
                    time_display = mod_time.strftime("%H:%M:%S")
                    parsed_time = mod_time
            except (ValueError, IndexError):
                # Fallback to file modification time if parsing fails
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                date_display = mod_time.strftime("%Y-%m-%d")
                time_display = mod_time.strftime("%H:%M:%S")
                parsed_time = mod_time
            
            # Check if transcript exists
            existing_transcript = self.transcript_exists(audio_file)
            
            metadata = {
                "name": audio_file.name,
                "path": audio_file,
                "date": date_display,
                "time": time_display,
                "modified": parsed_time,
                "duration": duration,
                "size_mb": stat.st_size / (1024 * 1024),
                "has_transcript": existing_transcript is not None,
                "transcript_path": existing_transcript
            }
            memos.append((audio_file, metadata))
        
        # Filter by date (ndays back) using parsed date from filename
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=self.ndays)
        memos = [(f, m) for f, m in memos if m['modified'] >= cutoff_time]
        
        # Sort by parsed date (newest first)
        memos.sort(key=lambda x: x[1]['modified'], reverse=True)
        
        # Limit to requested number if specified
        if limit is not None:
            memos = memos[:limit]
        
        return memos
    
    def select_voice_memo(self, auto_select_latest=False):
        """
        Interactive selection of voice memo to process.
        
        Args:
            auto_select_latest: If True, automatically select the latest memo
            
        Returns:
            Path to selected voice memo file
        """
        print(f"Searching for voice memos in: {self.voice_memos_dir}")
        print(f"  (recordings from past {self.ndays} days)")
        
        memos = self.list_voice_memos(limit=None)
        
        if not memos:
            print(f"Error: No audio files found in {self.voice_memos_dir} from the past {self.ndays} days")
            print(f"Looked for extensions: .m4a, .mp4, .m4v, .caf")
            print(f"Use --ndays to search further back (e.g., --ndays 30)")
            sys.exit(1)
        
        if auto_select_latest:
            selected_file = memos[0][0]
            metadata = memos[0][1]
            print(f"\nAuto-selected latest voice memo: {metadata['name']}")
            print(f"  Date: {metadata['date']} {metadata['time']}")
            if metadata['duration']:
                duration_str = self._format_duration(metadata['duration'])
                print(f"  Duration: {duration_str}")
            return selected_file
        
        # Display list
        print("\n" + "=" * 80)
        print(f"Voice Memos (past {self.ndays} days):")
        print("=" * 80)
        
        for i, (file_path, metadata) in enumerate(memos, 1):
            duration_str = self._format_duration(metadata['duration']) if metadata['duration'] else "unknown"
            transcript_indicator = " ✓" if metadata['has_transcript'] else ""
            
            print(f"{i:2d}. {metadata['date']} {metadata['time']}  [{duration_str:>8s}]{transcript_indicator}")
        
        print("=" * 80)
        if any(m[1]['has_transcript'] for m in memos):
            print("✓ = transcript already exists")
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nSelect memo [1-{len(memos)}, Enter for latest, or 'q' to quit]: ").strip().lower()
                
                if choice == 'q':
                    print("Cancelled.")
                    sys.exit(0)
                
                if choice == '':
                    # Empty input - select latest (first in list)
                    selected_file = memos[0][0]
                    selected_meta = memos[0][1]
                    print(f"\nSelected latest: {selected_meta['date']} {selected_meta['time']}")
                    return selected_file
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(memos):
                    selected_file = memos[choice_num - 1][0]
                    selected_meta = memos[choice_num - 1][1]
                    print(f"\nSelected: {selected_meta['date']} {selected_meta['time']}")
                    return selected_file
                else:
                    print(f"Please enter a number between 1 and {len(memos)}")
            except ValueError:
                print(f"Please enter a number between 1 and {len(memos)}, Enter for latest, or 'q' to quit")
            except KeyboardInterrupt:
                print("\n\nCancelled.")
                sys.exit(0)
    
    def _format_duration(self, seconds):
        """Format duration in seconds to MM:SS or HH:MM:SS string."""
        if seconds is None:
            return "unknown"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:2d}:{secs:02d}"
    
    def transcript_exists(self, audio_file):
        """
        Check if a transcript already exists for the given audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Path to existing transcript (SRT file) if found, None otherwise
        """
        audio_file = Path(audio_file)
        base_name = audio_file.stem
        
        # Ensure output_dir is absolute
        output_dir = self.output_dir
        if not output_dir.is_absolute():
            output_dir = Path.home() / output_dir
        
        # Look for any existing transcripts with this base name
        # Pattern: basename_YYYYMMDD_HHMMSS.srt
        existing_transcripts = list(output_dir.glob(f"{base_name}_*.srt"))
        
        if existing_transcripts:
            # Return the most recent transcript
            existing_transcripts.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return existing_transcripts[0]
        
        return None
    
    def find_oldest_unprocessed_memo(self):
        """
        Find the oldest voice memo from the past ndays that doesn't have a transcript.
        
        Returns:
            Path to oldest unprocessed audio file, or None if all are processed
        """
        from datetime import timedelta
        
        # Get all memos
        audio_extensions = [".m4a", ".mp4", ".m4v", ".caf"]
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.voice_memos_dir.glob(f"*{ext}"))
        
        if not audio_files:
            return None
        
        # Parse dates from filenames for accurate filtering
        files_with_dates = []
        for audio_file in audio_files:
            name = audio_file.stem
            try:
                # Try to parse timestamp from filename (format: YYYYMMDD HHMMSS-ID.ext)
                if len(name) >= 15 and name[8] == " ":
                    date_str = name[:8]  # YYYYMMDD
                    time_str = name[9:15]  # HHMMSS
                    parsed_time = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
                else:
                    # Fallback to file modification time
                    parsed_time = datetime.fromtimestamp(audio_file.stat().st_mtime)
            except (ValueError, IndexError):
                # Fallback to file modification time if parsing fails
                parsed_time = datetime.fromtimestamp(audio_file.stat().st_mtime)
            
            files_with_dates.append((audio_file, parsed_time))
        
        # Filter to only files from the past ndays
        cutoff_time = datetime.now() - timedelta(days=self.ndays)
        recent_files = [(f, d) for f, d in files_with_dates if d >= cutoff_time]
        
        # Sort by date, oldest first
        recent_files.sort(key=lambda x: x[1])
        
        # Find the first file without a transcript
        for audio_file, _ in recent_files:
            if self.transcript_exists(audio_file) is None:
                return audio_file
        
        return None
    
    def find_latest_voice_memo(self):
        """
        Find the most recently modified voice memo file.
        Deprecated: Use select_voice_memo() instead.
        """
        return self.select_voice_memo(auto_select_latest=True)
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe audio using selected backend.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcription result with segments
        """
        if self.backend == "parakeet":
            return self._transcribe_with_parakeet(audio_file)
        else:
            return self._transcribe_with_mlx_whisper(audio_file)
    
    def _transcribe_with_mlx_whisper(self, audio_file):
        """Transcribe using mlx-whisper."""
        global mlx_whisper
        if mlx_whisper is None:
            try:
                import mlx_whisper
            except ImportError:
                print("Error: mlx-whisper not installed.")
                print("Install with: pip install mlx-whisper")
                sys.exit(1)
        
        # Get model path
        if self.model_size not in self.WHISPER_MODELS:
            print(f"Warning: Unknown model size '{self.model_size}', using 'large-v3'")
            self.model_size = "large-v3"
        
        model_path = self.WHISPER_MODELS[self.model_size]
        
        print(f"\nTranscribing with mlx-whisper ({self.model_size})...")
        print(f"Model: {model_path}")
        print("This may take a few minutes depending on the length of the recording.")
        
        try:
            result = mlx_whisper.transcribe(
                str(audio_file),
                path_or_hf_repo=model_path,
                word_timestamps=True
            )
            
            print(f"Transcription complete. Found {len(result['segments'])} segments.")
            return result
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
    
    def _transcribe_with_parakeet(self, audio_file):
        """Transcribe using parakeet-mlx."""
        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            print("Error: parakeet-mlx not installed.")
            print("Install with: pip install parakeet-mlx")
            sys.exit(1)
        
        # Get model name
        if self.model_size not in self.PARAKEET_MODELS:
            print(f"Warning: Unknown model size '{self.model_size}' for parakeet, using 'medium'")
            self.model_size = "medium"
        
        model_name = self.PARAKEET_MODELS[self.model_size]
        
        print(f"\nTranscribing with parakeet-mlx ({self.model_size})...")
        print(f"Model: {model_name}")
        
        # Check audio duration to determine chunking strategy
        duration = self.get_audio_duration(audio_file)
        transcribe_kwargs = {}
        
        if duration:
            print(f"Recording duration: {self._format_duration(duration)}")
            if duration > 1200:  # > 20 minutes
                print("Using chunking for memory efficiency...")
                transcribe_kwargs["chunk_duration"] = 1200.0  # 20 minutes
                transcribe_kwargs["overlap_duration"] = 300.0  # 5 minutes
        
        print("This may take a few minutes depending on the length of the recording.")
        
        try:
            # Load model using correct API
            print("Loading model...")
            model = from_pretrained(model_name)
            
            # Always use local attention to prevent GPU hangs
            print("Configuring local attention...")
            model.encoder.set_attention_model(
                "rel_pos_local_attn",  # NeMo's naming convention
                (256, 256)  # (left_context, right_context) frames
            )
            print("Local attention enabled.")
            
            # Transcribe with appropriate strategy
            print("Transcribing...")
            if transcribe_kwargs:
                chunk_mins = transcribe_kwargs.get('chunk_duration', 0) / 60
                overlap_mins = transcribe_kwargs.get('overlap_duration', 0) / 60
                
                # Calculate expected number of chunks for progress display
                if duration:
                    chunk_duration = transcribe_kwargs.get('chunk_duration', 1200)
                    overlap_duration = transcribe_kwargs.get('overlap_duration', 300)
                    expected_chunks = max(1, int((duration - overlap_duration) / (chunk_duration - overlap_duration)) + 1)
                else:
                    expected_chunks = None
                
                print(f"Chunking enabled: {chunk_mins:.1f} min chunks, {overlap_mins:.1f} min overlap")
                if expected_chunks:
                    print(f"Estimated chunks: {expected_chunks}")
                
                # Add progress callback for chunked transcription
                def progress_callback(current_samples, total_samples):
                    """Report progress during chunked transcription."""
                    if expected_chunks and total_samples > 0:
                        # Estimate chunk number from sample progress
                        progress_pct = (current_samples / total_samples) * 100
                        current_chunk = int((current_samples / total_samples) * expected_chunks) + 1
                        current_chunk = min(current_chunk, expected_chunks)  # Cap at expected
                        print(f"  Processing chunk {current_chunk}/{expected_chunks} ({progress_pct:.0f}%)...", end="\r")
                        if current_samples >= total_samples:
                            print()  # Newline when complete
                    else:
                        # Fallback to percentage if we can't estimate chunks
                        progress_pct = (current_samples / total_samples * 100) if total_samples > 0 else 0
                        print(f"  Processing: {progress_pct:.0f}%...", end="\r")
                        if current_samples >= total_samples:
                            print()
                
                transcribe_kwargs["chunk_callback"] = progress_callback
            
            result = model.transcribe(str(audio_file), **transcribe_kwargs)
            
            # Convert parakeet result format to whisper-like format with segments
            # Parakeet returns result.sentences with AlignedSentence objects
            segments = []
            
            if hasattr(result, 'sentences') and result.sentences:
                # Parakeet provides sentences with timestamps
                print(f"Found {len(result.sentences)} sentences from parakeet")
                for sentence in result.sentences:
                    segments.append({
                        'start': sentence.start,
                        'end': sentence.end,
                        'text': sentence.text
                    })
            elif hasattr(result, 'text'):
                # Fallback: only text is available, create a single segment
                print("Warning: Only text available, no timestamps")
                segments = [{
                    'start': 0.0,
                    'end': 0.0,
                    'text': result.text
                }]
            else:
                # Fallback: treat result as text
                print("Warning: Unknown result format")
                segments = [{
                    'start': 0.0,
                    'end': 0.0,
                    'text': str(result)
                }]
            
            # Return in whisper-compatible format
            whisper_format = {
                'text': result.text if hasattr(result, 'text') else str(result),
                'segments': segments
            }
            
            print(f"Transcription complete. Created {len(segments)} segments.")
            return whisper_format
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def diarize_audio(self, audio_file):
        """
        Perform speaker diarization using pyannote.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Diarization result with speaker segments
        """
        # Load heavy imports if not already loaded
        load_heavy_imports()
        
        print(f"\nPerforming speaker diarization...")
        print("Note: This requires a Hugging Face token with access to pyannote models.")
        
        # Check for HF token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            print("\nWarning: No Hugging Face token found.")
            print("To use speaker diarization, you need to:")
            print("1. Create an account at https://huggingface.co")
            print("2. Accept terms at:")
            print("   - https://huggingface.co/pyannote/speaker-diarization-community-1 (primary)")
            print("   - https://huggingface.co/pyannote/segmentation-3.0")
            print("3. Set your token: export HF_TOKEN='your_token_here'")
            print("\nSkipping diarization for now...")
            return None
        
        try:
            # Handle PyTorch 2.6+ weights_only change
            # Temporarily override torch.load to use weights_only=False for trusted pyannote models
            import torch
            
            # Save original torch.load
            _original_torch_load = torch.load
            
            def _patched_torch_load(*args, **kwargs):
                # Force weights_only=False for pyannote model loading
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            # Apply patch
            torch.load = _patched_torch_load
            
            try:
                # Use community-1 model which is ~40% faster (2.5% real-time factor)
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=hf_token
                )
            finally:
                # Restore original torch.load
                torch.load = _original_torch_load
            
            # Run on CPU or MPS (Metal Performance Shaders for Apple Silicon)
            if torch.backends.mps.is_available():
                pipeline.to(torch.device("mps"))
                print("Using Apple Silicon GPU (MPS) for diarization")
            
            # Optimization: Reduce embedding batch size (counter-intuitive but faster)
            pipeline.embedding_batch_size = 1
            
            # Optimization: Pre-load audio into memory for faster processing
            print("Pre-loading audio...")
            import torchaudio
            load_start = time.time()
            waveform, sample_rate = torchaudio.load(str(audio_file))
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            load_time = time.time() - load_start
            print(f"  Audio loaded in {load_time:.1f}s")
            
            # Prepare diarization parameters
            diarization_kwargs = {}
            if self.num_speakers is not None:
                print(f"Using known speaker count: {self.num_speakers} (faster processing)")
                diarization_kwargs["num_speakers"] = self.num_speakers
            
            # Run diarization with progress hook
            print("\nDiarization progress:")
            diarization_start = time.time()
            
            with ProgressHook() as hook:
                diarization = pipeline(audio_input, hook=hook, **diarization_kwargs)
            
            diarization_time = time.time() - diarization_start
            
            print(f"✓ Diarization complete in {diarization_time:.1f}s")
            return diarization
            
        except Exception as e:
            print(f"Error during diarization: {e}")
            print("Continuing without speaker labels...")
            return None
    
    def merge_transcription_and_diarization(self, transcription, diarization, audio_filename=""):
        """
        Merge transcription segments with speaker labels from diarization.
        
        Args:
            transcription: Whisper transcription result
            diarization: pyannote diarization result (DiarizeOutput or Annotation)
            audio_filename: Name of audio file for speaker database
            
        Returns:
            List of segments with text and speaker labels
        """
        if diarization is None:
            # No diarization, just return transcription segments
            return [{
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": "Unknown"
            } for seg in transcription["segments"]]
        
        # Handle DiarizeOutput wrapper (newer pyannote versions)
        # DiarizeOutput has speaker_diarization, exclusive_speaker_diarization, speaker_embeddings
        annotation = diarization
        embeddings = None
        
        if hasattr(diarization, 'speaker_diarization'):
            # Newer API: DiarizeOutput wrapper
            annotation = diarization.speaker_diarization
            embeddings = diarization.speaker_embeddings
            print(f"Extracted speaker_diarization from DiarizeOutput")
        elif not hasattr(diarization, 'itertracks'):
            # Fallback: try other attributes
            if hasattr(diarization, 'annotation'):
                annotation = diarization.annotation
            elif hasattr(diarization, 'annotations'):
                annotation = list(diarization.annotations.values())[0]
        
        # Get list of unique speakers
        speaker_labels = list(annotation.labels())
        
        # Try to identify speakers using database
        speaker_mapping = {}
        if self.use_speaker_db and embeddings is not None and len(speaker_labels) > 0:
            # Initialize speaker database on first use
            if self.speaker_db is None:
                self.speaker_db = SpeakerDatabase()
            
            print(f"\nIdentifying {len(speaker_labels)} speaker(s)...")
            
            identified = self.speaker_db.identify_speakers(
                embeddings, 
                speaker_labels,
                threshold=0.75
            )
            
            # Get multiple sample quotes for each speaker to help with identification
            speaker_quotes = {}  # speaker_label -> list of quotes
            speaker_audio_segments = {}  # speaker_label -> list of (start, end) tuples for audio clips
            
            for segment in transcription["segments"]:
                seg_start = segment["start"]
                seg_end = segment["end"]
                seg_mid = (seg_start + seg_end) / 2
                
                # Find which speaker this segment belongs to
                for turn, _, speaker_label in annotation.itertracks(yield_label=True):
                    if turn.start <= seg_mid <= turn.end:
                        # Collect multiple quotes per speaker (up to 5)
                        if speaker_label not in speaker_quotes:
                            speaker_quotes[speaker_label] = []
                        if len(speaker_quotes[speaker_label]) < 5:
                            quote = segment["text"].strip()
                            if quote and len(quote) > 10:  # Skip very short fragments
                                speaker_quotes[speaker_label].append(quote)
                        
                        # Store multiple good audio segments for playback (up to 3)
                        if speaker_label not in speaker_audio_segments:
                            speaker_audio_segments[speaker_label] = []
                        if len(speaker_audio_segments[speaker_label]) < 3:
                            duration = seg_end - seg_start
                            if 2.0 <= duration <= 8.0:  # Good length for identification
                                speaker_audio_segments[speaker_label].append((seg_start, seg_end))
                        break
            
            # Prompt for names of unknown speakers
            # Track user interaction time separately
            def track_interaction_time(duration):
                if not hasattr(self, '_user_interaction_time'):
                    self._user_interaction_time = 0.0
                self._user_interaction_time += duration
            
            # Pass audio file to speaker database for playback
            if hasattr(self, '_current_audio_file'):
                self.speaker_db._audio_file_for_playback = self._current_audio_file
            
            speaker_mapping = self.speaker_db.prompt_for_speaker_names(
                embeddings,
                speaker_labels,
                identified,
                audio_filename,
                speaker_quotes,
                time_tracker=track_interaction_time,
                audio_segments=speaker_audio_segments,
                confidence_threshold=self.speaker_confidence_threshold
            )
        else:
            # No speaker recognition - use original labels
            speaker_mapping = {label: label for label in speaker_labels}
        
        # Now merge transcription with identified speakers
        merged_segments = []
        
        for segment in transcription["segments"]:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2
            
            # Find the speaker at the middle of this segment
            speaker = "Unknown"
            
            try:
                # Standard pyannote Annotation API
                for turn, _, speaker_label in annotation.itertracks(yield_label=True):
                    if turn.start <= seg_mid <= turn.end:
                        # Map to identified name
                        speaker = speaker_mapping.get(speaker_label, speaker_label)
                        break
            except Exception as e:
                print(f"Warning: Could not match speaker for segment at {seg_mid}s: {e}")
            
            merged_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": segment["text"],
                "speaker": speaker
            })
        
        return merged_segments
    
    def format_time(self, seconds):
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def save_srt(self, segments, output_path):
        """
        Save segments as an SRT subtitle file.
        
        Args:
            segments: List of segment dictionaries with start, end, text, speaker
            output_path: Path to save SRT file
        """
        print(f"\nSaving transcript to: {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                # SRT format:
                # 1
                # 00:00:00,000 --> 00:00:04,000
                # [Speaker A] Text here
                f.write(f"{i}\n")
                f.write(f"{self.format_time(seg['start'])} --> {self.format_time(seg['end'])}\n")
                f.write(f"[{seg['speaker']}] {seg['text'].strip()}\n")
                f.write("\n")
        
        print(f"Transcript saved successfully.")
    
    def save_json(self, segments, output_path):
        """Save segments as JSON for easier processing."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
    
    def generate_summary_with_cursor(self, srt_path, workspace_path=None):
        """
        Use Cursor CLI to generate a summary.
        
        Args:
            srt_path: Path to the SRT transcript
            workspace_path: Path to Cursor workspace (for --workspace flag)
        """
        print(f"\nGenerating summary with Cursor...")
        
        prompt = f"""Please create a summary of the transcription at: {srt_path}"""
        
        try:
            # Try to find cursor CLI command (try both 'cursor-agent' and 'agent')
            cursor_cmd = None
            for cmd_name in ["cursor-agent", "agent"]:
                try:
                    # Check if command exists
                    subprocess.run([cmd_name, "--version"], 
                                 capture_output=True, 
                                 check=False,
                                 timeout=5)
                    cursor_cmd = cmd_name
                    break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if not cursor_cmd:
                raise FileNotFoundError("Neither 'cursor-agent' nor 'agent' command found")
            
            # Build command with flags
            cmd = [cursor_cmd, "--force"]
            
            # Add workspace path if provided
            if workspace_path:
                cmd.extend(["--workspace", str(workspace_path)])
            
            # Add prompt
            cmd.extend(["-p", prompt])
            
            # Execute cursor CLI with flags
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print("✓ Summary generation initiated with Cursor")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"Warning: {cursor_cmd} returned error code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                    # Check for common authentication errors
                    if "401" in result.stderr or "unauthorized" in result.stderr.lower() or "authentication" in result.stderr.lower():
                        print("\nAuthentication error detected!")
                        print("Make sure CURSOR_API_KEY is set in your environment.")
                        print("Get your API key from: https://cursor.com/dashboard?tab=cloud-agents")
                        print("Set it with: export CURSOR_API_KEY='your_key_here'")
                print("\nFallback: Please manually create the summary using:")
                print(f"  Transcript: {srt_path}")
        
        except subprocess.TimeoutExpired:
            print("Warning: Cursor command timed out")
        except FileNotFoundError:
            print("Warning: Cursor CLI not found")
            print("Install Cursor CLI from: https://cursor.com/docs/cli/installation")
            print("Make sure it's in your PATH after installation")
            print("\nFallback: Please manually create the summary using:")
            print(f"  Transcript: {srt_path}")
        except Exception as e:
            print(f"Error invoking Cursor: {e}")
            print("\nFallback: Please manually create the summary using:")
            print(f"  Transcript: {srt_path}")
    
    def process_voice_memo(self, audio_file=None, auto_select_latest=False,
                           generate_summary=True, workspace_path=None, force_overwrite=False,
                           skip_prompt=False):
        """
        Main processing pipeline.
        
        Args:
            audio_file: Specific audio file to process (if None, prompts for selection)
            auto_select_latest: If True, automatically select latest memo
            generate_summary: Whether to generate a summary with Cursor
            workspace_path: Path to Cursor workspace (for --workspace flag)
            force_overwrite: If True, process even if transcript exists (used with --latest --force)
            skip_prompt: If True, skip prompting about existing transcripts (for --auto mode)
        """
        print(f"Configuration:")
        print(f"  Backend: {self.backend}")
        print(f"  Model size: {self.model_size}")
        print()
        
        # Find or select voice memo
        if audio_file is None:
            audio_file = self.select_voice_memo(auto_select_latest=auto_select_latest)
            
            # If interactive mode and num_speakers not specified, prompt for it
            if not auto_select_latest and self.num_speakers is None:
                try:
                    response = input("\nHow many speakers? (press Enter to skip): ").strip()
                    if response:
                        num_speakers = int(response)
                        if num_speakers > 0:
                            self.num_speakers = num_speakers
                            print(f"Will optimize diarization for {num_speakers} speaker(s)")
                except (ValueError, KeyboardInterrupt):
                    print("Skipping speaker count optimization")
        else:
            audio_file = Path(audio_file)
            if not audio_file.exists():
                print(f"Error: File not found: {audio_file}")
                sys.exit(1)
            print(f"Processing: {audio_file.name}")
            
            # Prompt for number of speakers if not specified (skip in auto mode)
            if self.num_speakers is None and not skip_prompt:
                try:
                    response = input("\nHow many speakers? (press Enter to skip): ").strip()
                    if response:
                        num_speakers = int(response)
                        if num_speakers > 0:
                            self.num_speakers = num_speakers
                            print(f"Will optimize diarization for {num_speakers} speaker(s)")
                except (ValueError, KeyboardInterrupt):
                    print("Skipping speaker count optimization")
        
        # Check if transcript already exists
        existing_transcript = self.transcript_exists(audio_file)
        if existing_transcript:
            print(f"\n⚠️  Transcript already exists: {existing_transcript}")
            
            if skip_prompt:
                # For --auto mode, silently skip files that already have transcripts
                print("Skipping (already processed).")
                return None
            elif auto_select_latest and not force_overwrite:
                # For --latest without --force, skip without prompting
                print("Use --force to overwrite existing transcript.")
                print("Skipping processing.")
                return None
            elif not auto_select_latest and not force_overwrite:
                # Interactive mode: prompt user
                try:
                    response = input("Overwrite existing transcript? [y/N]: ").strip().lower()
                    if response not in ['y', 'yes']:
                        print("Skipping processing.")
                        return None
                    print("Proceeding with transcription...")
                except KeyboardInterrupt:
                    print("\n\nCancelled.")
                    return None
            # If force_overwrite is True, proceed without prompting
        
        # Now load heavy dependencies after user has made their selection
        print("\nLoading AI models...")
        load_heavy_imports()
        
        # Create output filename based on audio filename
        base_name = audio_file.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output_dir is absolute
        output_dir = self.output_dir
        if not output_dir.is_absolute():
            output_dir = Path.home() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_base = output_dir / f"{base_name}_{timestamp}"
        
        # Track timing for all steps
        timings = {}
        overall_start = time.time()
        
        # Transcribe
        print("\n" + "="*60)
        print("STEP 1: Transcription")
        print("="*60)
        transcribe_start = time.time()
        transcription = self.transcribe_audio(audio_file)
        timings['transcription'] = time.time() - transcribe_start
        print(f"✓ Transcription completed in {timings['transcription']:.1f}s")
        
        # Diarize
        print("\n" + "="*60)
        print("STEP 2: Speaker Diarization")
        print("="*60)
        diarize_start = time.time()
        diarization = self.diarize_audio(audio_file)
        timings['diarization'] = time.time() - diarize_start
        
        # Merge (with speaker recognition if enabled)
        print("\n" + "="*60)
        print("STEP 3: Merging & Speaker Recognition")
        print("="*60)
        merge_start = time.time()
        
        # Pause timing during user interaction
        if hasattr(self, '_user_interaction_time'):
            self._user_interaction_time = 0.0
        
        # Store audio file path for speaker identification playback
        self._current_audio_file = audio_file
        
        segments = self.merge_transcription_and_diarization(
            transcription, 
            diarization, 
            audio_filename=base_name
        )
        
        # Subtract user interaction time from merge timing
        merge_time = time.time() - merge_start
        if hasattr(self, '_user_interaction_time'):
            merge_time -= self._user_interaction_time
        timings['merge'] = merge_time
        print(f"✓ Merge completed in {timings['merge']:.1f}s")
        
        # Save outputs
        print("\n" + "="*60)
        print("STEP 4: Saving Output")
        print("="*60)
        save_start = time.time()
        srt_path = output_base.with_suffix(".srt")
        json_path = output_base.with_suffix(".json")
        
        self.save_srt(segments, srt_path)
        self.save_json(segments, json_path)
        timings['save'] = time.time() - save_start
        print(f"✓ Save completed in {timings['save']:.1f}s")
        
        print("\n✓ Processing complete!")
        print(f"  SRT file: {srt_path}")
        print(f"  JSON file: {json_path}")
        
        # Generate summary if requested
        if generate_summary:
            print("\n" + "="*60)
            print("STEP 5: Summary Generation")
            print("="*60)
            summary_start = time.time()
            self.generate_summary_with_cursor(srt_path, workspace_path)
            timings['summary'] = time.time() - summary_start
            print(f"✓ Summary completed in {timings['summary']:.1f}s")
        
        # Print timing summary
        timings['total'] = time.time() - overall_start
        
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        audio_duration = self.get_audio_duration(audio_file)
        if audio_duration:
            print(f"Audio duration: {self._format_duration(audio_duration)}")
        print(f"Transcription: {timings['transcription']:.1f}s")
        print(f"Diarization:   {timings['diarization']:.1f}s")
        print(f"Merge/Speaker: {timings['merge']:.1f}s")
        print(f"Save files:    {timings['save']:.1f}s")
        if 'summary' in timings:
            print(f"Summary:       {timings['summary']:.1f}s")
        print(f"Total:         {timings['total']:.1f}s")
        if audio_duration:
            processing_time = timings['transcription'] + timings['diarization']
            realtime_factor = processing_time / audio_duration
            print(f"Real-time factor: {realtime_factor:.2f}x")
        
        return srt_path


def main():
    """Main entry point."""
    # Try to load config file
    config_defaults = {
        "DEFAULT_BACKEND": "parakeet",
        "DEFAULT_MODEL_SIZE": "medium",
        "VOICE_MEMOS_DIR": None,
        "OUTPUT_DIR": None,
        "CURSOR_WORKSPACE_PATH": None,
        "GENERATE_SUMMARY": True,
        "SPEAKER_CONFIDENCE_THRESHOLD": 0.95
    }
    
    try:
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            for key in config_defaults:
                if hasattr(config, key):
                    config_defaults[key] = getattr(config, key)
    except Exception:
        pass  # Use defaults if config fails to load
    
    parser = argparse.ArgumentParser(
        description="Transcribe and diarize Apple Voice Memos with local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive selection (default - shows recordings from past 7 days)
  %(prog)s
  
  # Show recordings from past 30 days
  %(prog)s --ndays 30
  
  # Auto-select latest memo without prompt
  %(prog)s --latest
  
  # Force re-process latest memo even if transcript exists
  %(prog)s --latest --force
  
  # Automatic mode: process oldest unprocessed memo (for cron jobs)
  %(prog)s --auto
  
  # Auto mode searching past 14 days
  %(prog)s --auto --ndays 14
  
  # Use mlx-whisper with large model for best accuracy
  %(prog)s --backend mlx-whisper --model large-v3 --latest
  
  # Process a specific file
  %(prog)s --file /path/to/recording.m4a
  
  # Quick transcription with small model
  %(prog)s --backend parakeet --model small --no-summary --latest
        """
    )
    parser.add_argument(
        "--voice-memos-dir",
        type=Path,
        default=config_defaults["VOICE_MEMOS_DIR"],
        help="Path to Voice Memos recordings folder (default: standard Apple location)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config_defaults["OUTPUT_DIR"],
        help="Directory to save transcripts (default: ~/transcripts)"
    )
    parser.add_argument(
        "--backend",
        choices=["mlx-whisper", "parakeet"],
        default=config_defaults["DEFAULT_BACKEND"],
        help=f"Transcription backend (default: {config_defaults['DEFAULT_BACKEND']}, faster processing)"
    )
    parser.add_argument(
        "--model",
        default=config_defaults["DEFAULT_MODEL_SIZE"],
        help=f"Model size: tiny/base/small/medium/large/large-v3/large-v3-turbo (default: {config_defaults['DEFAULT_MODEL_SIZE']})"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating summary with Cursor"
    )
    parser.add_argument(
        "--no-speaker-db",
        action="store_true",
        help="Disable speaker recognition database"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Automatically select the latest voice memo without prompting"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process any audio file (m4a, wav, mp3, etc.) instead of selecting from Voice Memos"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers in recording (speeds up diarization significantly if known)"
    )
    parser.add_argument(
        "--ndays",
        type=int,
        default=7,
        help="Number of days back to search for recordings (default: 7). Affects interactive list and --auto mode"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mode: process oldest unprocessed memo from past ndays (for cron jobs). No prompts, no overwrites. Incompatible with --force, --latest, --file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing transcripts (use with --latest). Incompatible with --auto"
    )
    parser.add_argument(
        "--speaker-confidence-threshold",
        type=float,
        default=config_defaults.get("SPEAKER_CONFIDENCE_THRESHOLD", 0.95),
        help="Only prompt for speaker confirmation if confidence < threshold (default: 0.95)"
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.auto and args.force:
        parser.error("--auto and --force are incompatible (--auto is designed for unattended operation)")
    
    if args.auto and args.latest:
        parser.error("--auto and --latest are incompatible (use one or the other)")
    
    if args.auto and args.file:
        parser.error("--auto cannot be used with --file (--auto searches for unprocessed memos)")
    
    # Allow config to override summary default
    generate_summary = config_defaults["GENERATE_SUMMARY"] and not args.no_summary
    
    # Create transcriber
    transcriber = VoiceMemoTranscriber(
        voice_memos_dir=args.voice_memos_dir,
        output_dir=args.output_dir,
        backend=args.backend,
        model_size=args.model,
        use_speaker_db=not args.no_speaker_db,
        num_speakers=args.num_speakers,
        speaker_confidence_threshold=args.speaker_confidence_threshold,
        ndays=args.ndays
    )
    
    # Handle --auto mode
    if args.auto:
        print(f"Auto mode: searching for oldest unprocessed memo from past {args.ndays} days...")
        audio_file = transcriber.find_oldest_unprocessed_memo()
        if audio_file is None:
            print(f"✓ No unprocessed memos found. All recordings from the past {args.ndays} days have been transcribed.")
            sys.exit(0)
        
        print(f"Found unprocessed memo: {audio_file.name}")
        
        # Get metadata for display
        stat = audio_file.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        date_display = mod_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = transcriber.get_audio_duration(audio_file)
        duration_str = transcriber._format_duration(duration) if duration else "unknown"
        print(f"  Date: {date_display}")
        print(f"  Duration: {duration_str}")
        
        # Process with skip_prompt=True for auto mode
        result = transcriber.process_voice_memo(
            audio_file=audio_file,
            auto_select_latest=False,
            generate_summary=generate_summary,
            workspace_path=config_defaults["CURSOR_WORKSPACE_PATH"],
            force_overwrite=False,
            skip_prompt=True
        )
        
        if result is None:
            print("Skipped (already processed)")
            sys.exit(0)
    else:
        # Normal mode
        transcriber.process_voice_memo(
            audio_file=args.file,
            auto_select_latest=args.latest,
            generate_summary=generate_summary,
            workspace_path=config_defaults["CURSOR_WORKSPACE_PATH"],
            force_overwrite=args.force,
            skip_prompt=False
        )


if __name__ == "__main__":
    main()
