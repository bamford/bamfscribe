"""
Speaker Recognition Database

Manages a database of known speaker voice profiles (embeddings) for
automatic speaker identification across multiple recordings.
"""

import json
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SpeakerDatabase:
    """Manages speaker profiles for automatic recognition."""
    
    def __init__(self, database_path: Path = None):
        """
        Initialize speaker database.
        
        Args:
            database_path: Path to JSON file storing speaker profiles
                          (defaults to ~/speaker_profiles.json)
        """
        if database_path is None:
            database_path = Path.home() / ".speaker_profiles.json"
        
        self.database_path = Path(database_path)
        self.speakers = self.load_database()
    
    def load_database(self) -> Dict:
        """Load speaker database from disk."""
        if not self.database_path.exists():
            return {}
        
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            # Convert embedding lists back to numpy arrays
            for speaker in data.values():
                speaker['embedding'] = np.array(speaker['embedding'])
            
            return data
        except Exception as e:
            print(f"Warning: Could not load speaker database: {e}")
            return {}
    
    def save_database(self):
        """Save speaker database to disk."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for name, data in self.speakers.items():
            serializable_data[name] = {
                'embedding': data['embedding'].tolist(),
                'recordings_count': data.get('recordings_count', 1),
                'last_seen': data.get('last_seen', '')
            }
        
        with open(self.database_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def add_speaker(self, name: str, embedding: np.ndarray, recording_name: str = ""):
        """
        Add or update a speaker profile.
        
        Args:
            name: Speaker name
            embedding: Speaker voice embedding vector
            recording_name: Name of the recording (for tracking)
        """
        if name in self.speakers:
            # Update existing speaker - average the embeddings
            old_embedding = self.speakers[name]['embedding']
            count = self.speakers[name].get('recordings_count', 1)
            
            # Weighted average to gradually adapt to voice changes
            new_embedding = (old_embedding * count + embedding) / (count + 1)
            
            self.speakers[name]['embedding'] = new_embedding
            self.speakers[name]['recordings_count'] = count + 1
            self.speakers[name]['last_seen'] = recording_name
            
            print(f"Updated profile for {name} (seen in {count + 1} recordings)")
        else:
            # Add new speaker
            self.speakers[name] = {
                'embedding': embedding,
                'recordings_count': 1,
                'last_seen': recording_name
            }
            print(f"Added new speaker profile: {name}")
        
        self.save_database()
    
    def identify_speakers(
        self, 
        embeddings: np.ndarray, 
        speaker_labels: List[str],
        threshold: float = 0.75
    ) -> Dict[str, Tuple[Optional[str], float]]:
        """
        Identify speakers by comparing embeddings to database.
        
        Args:
            embeddings: (num_speakers, dimension) array from diarization
            speaker_labels: List of speaker IDs (e.g., ["SPEAKER_00", "SPEAKER_01"])
            threshold: Minimum similarity score (0-1) to consider a match
        
        Returns:
            Dictionary mapping speaker_label -> (identified_name, confidence)
            where identified_name is None if no match found
        """
        if len(self.speakers) == 0:
            print("No speakers in database yet.")
            return {label: (None, 0.0) for label in speaker_labels}
        
        # Build database embedding matrix
        db_names = list(self.speakers.keys())
        db_embeddings = np.array([self.speakers[name]['embedding'] for name in db_names])
        
        # Compute similarity between each new speaker and all known speakers
        results = {}
        
        for i, label in enumerate(speaker_labels):
            speaker_embedding = embeddings[i:i+1]  # Keep 2D shape
            
            # Compute cosine similarity with all known speakers
            similarities = cosine_similarity(speaker_embedding, db_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score >= threshold:
                best_name = db_names[best_idx]
                results[label] = (best_name, float(best_score))
                print(f"{label} → {best_name} (confidence: {best_score:.2%})")
            else:
                results[label] = (None, float(best_score))
                print(f"{label} → Unknown (best match: {db_names[best_idx]}, confidence: {best_score:.2%})")
        
        return results
    
    def prompt_for_speaker_names(
        self,
        embeddings: np.ndarray,
        speaker_labels: List[str],
        identified_speakers: Dict[str, Tuple[Optional[str], float]],
        recording_name: str = "",
        speaker_quotes: Dict[str, List[str]] = None,
        time_tracker: callable = None,
        audio_segments: Dict[str, Tuple[float, float]] = None,
        confidence_threshold: float = 0.95
    ) -> Dict[str, str]:
        """
        Interactively prompt user to name unknown speakers.
        
        Args:
            embeddings: Speaker embeddings array
            speaker_labels: List of speaker IDs
            identified_speakers: Results from identify_speakers()
            recording_name: Name of current recording
            speaker_quotes: Dictionary mapping speaker_label -> list of sample quotes
            time_tracker: Optional callback to track user interaction time
            audio_segments: Dictionary mapping speaker_label -> list of (start, end) tuples
            confidence_threshold: Only prompt for confirmation if confidence < threshold (default: 0.95)
        
        Returns:
            Dictionary mapping speaker_label -> final_name
        """
        
        # Track which audio clip index we're on for each speaker
        audio_clip_indices = {}
        
        def play_speaker_audio(label):
            """Play audio clip for a speaker (cycles through available clips)."""
            if not audio_segments or label not in audio_segments:
                print("    [Audio not available]")
                return
            
            clips = audio_segments[label]
            if not clips:
                print("    [Audio not available]")
                return
            
            try:
                import torchaudio
                from pathlib import Path
                
                # Get the audio file from the transcriber instance
                audio_file = getattr(self, '_audio_file_for_playback', None)
                if not audio_file:
                    print("    [Audio file not available]")
                    return
                
                # Get the current clip index for this speaker (cycling through available clips)
                if label not in audio_clip_indices:
                    audio_clip_indices[label] = 0
                clip_idx = audio_clip_indices[label]
                
                start, end = clips[clip_idx]
                
                # Advance to next clip for next time (wrap around)
                audio_clip_indices[label] = (clip_idx + 1) % len(clips)
                
                # Load and extract the segment
                waveform, sample_rate = torchaudio.load(str(audio_file))
                start_frame = int(start * sample_rate)
                end_frame = int(end * sample_rate)
                clip = waveform[:, start_frame:end_frame]
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    torchaudio.save(tmp_path, clip, sample_rate)
                
                # Play using afplay (macOS)
                clip_info = f"clip {clip_idx + 1}/{len(clips)}, {end - start:.1f}s" if len(clips) > 1 else f"{end - start:.1f}s"
                print(f"    [Playing audio {clip_info}...]")
                subprocess.run(['afplay', tmp_path], check=True)
                
                # Clean up
                Path(tmp_path).unlink()
                
            except Exception as e:
                print(f"    [Could not play audio: {e}]")
        final_names = {}
        
        print("\n" + "="*60)
        print("Speaker Identification")
        print("="*60)
        
        for i, label in enumerate(speaker_labels):
            identified_name, confidence = identified_speakers[label]
            
            # Show a sample quote to help identify the speaker
            quotes = speaker_quotes.get(label, []) if speaker_quotes else []
            if quotes:
                quote = quotes[0]
                # Truncate long quotes
                if len(quote) > 80:
                    quote = quote[:77] + "..."
                print(f"\n{label}: \"{quote}\"")
            
            if identified_name is not None:
                # Confirmed match
                print(f"  Identified as '{identified_name}' ({confidence:.2%} confidence)")
                
                # Auto-confirm if confidence is above threshold
                if confidence >= confidence_threshold:
                    print(f"  Auto-confirmed (confidence >= {confidence_threshold:.0%})")
                    final_names[label] = identified_name
                    # Update the profile
                    self.add_speaker(identified_name, embeddings[i], recording_name)
                else:
                    # Confidence below threshold - prompt for confirmation
                    # Track time spent waiting for user input
                    input_start = time.time() if time_tracker else None
                    prompt = f"  Confirm (y), rename (n), skip (s)"
                    options = "y/n/s"
                    if len(quotes) > 1:
                        prompt += f", more quotes (m)"
                        options += "/m"
                    if audio_segments and label in audio_segments:
                        prompt += f", play audio (p)"
                        options += "/p"
                    prompt += f"? [{options}]: "
                    response = input(prompt).strip().lower()
                    if time_tracker and input_start:
                        time_tracker(time.time() - input_start)
                    
                    # Keep handling commands until we get a decision
                    while True:
                        # Handle "play audio" request
                        if response == 'p':
                            play_speaker_audio(label)
                            input_start = time.time() if time_tracker else None
                            response = input(prompt).strip().lower()
                            if time_tracker and input_start:
                                time_tracker(time.time() - input_start)
                            continue
                        
                        # Handle "more quotes" request
                        if response == 'm' and len(quotes) > 1:
                            print(f"\n  Additional quotes from {label}:")
                            for idx, q in enumerate(quotes[1:], 2):
                                display_q = q[:77] + "..." if len(q) > 80 else q
                                print(f"    {idx}. \"{display_q}\"")
                            
                            input_start = time.time() if time_tracker else None
                            response = input(prompt).strip().lower()
                            if time_tracker and input_start:
                                time_tracker(time.time() - input_start)
                            continue
                        
                        # Not a command, break out to handle decision
                        break
                    
                    if response == 'n':
                        # User wants to rename
                        input_start = time.time() if time_tracker else None
                        new_name = input(f"  Enter correct name: ").strip()
                        if time_tracker and input_start:
                            time_tracker(time.time() - input_start)
                        if new_name:
                            final_names[label] = new_name
                            # Add/update with correct name
                            self.add_speaker(new_name, embeddings[i], recording_name)
                    elif response == 's':
                        # Skip - keep original label
                        final_names[label] = label
                    else:
                        # Confirmed
                        final_names[label] = identified_name
                        # Update the profile
                        self.add_speaker(identified_name, embeddings[i], recording_name)
            else:
                # New/unknown speaker
                print(f"  Unknown speaker detected")
                
                input_start = time.time() if time_tracker else None
                prompt = f"  Enter name"
                options = []
                if len(quotes) > 1:
                    options.append("more quotes (m)")
                if audio_segments and label in audio_segments:
                    options.append("play audio (p)")
                if options:
                    prompt += f", {', '.join(options)}, or press Enter to skip"
                else:
                    prompt += f" (or press Enter to skip)"
                prompt += ": "
                response = input(prompt).strip()
                if time_tracker and input_start:
                    time_tracker(time.time() - input_start)
                
                # Keep handling commands until we get a name or skip
                while True:
                    # Handle "play audio" request
                    if response.lower() == 'p':
                        play_speaker_audio(label)
                        input_start = time.time() if time_tracker else None
                        response = input(prompt).strip()
                        if time_tracker and input_start:
                            time_tracker(time.time() - input_start)
                        continue
                    
                    # Handle "more quotes" request
                    if response.lower() == 'm' and len(quotes) > 1:
                        print(f"\n  Additional quotes from {label}:")
                        for idx, q in enumerate(quotes[1:], 2):
                            display_q = q[:77] + "..." if len(q) > 80 else q
                            print(f"    {idx}. \"{display_q}\"")
                        
                        input_start = time.time() if time_tracker else None
                        response = input(prompt).strip()
                        if time_tracker and input_start:
                            time_tracker(time.time() - input_start)
                        continue
                    
                    # Not a command, break out
                    break
                
                if response:
                    final_names[label] = response
                    self.add_speaker(response, embeddings[i], recording_name)
                else:
                    final_names[label] = label
        
        print("\n" + "="*60)
        return final_names
    
    def list_speakers(self):
        """Print all known speakers."""
        if not self.speakers:
            print("No speakers in database.")
            return
        
        print("\nKnown Speakers:")
        print("-" * 60)
        for name, data in sorted(self.speakers.items()):
            count = data.get('recordings_count', 1)
            last_seen = data.get('last_seen', 'Unknown')
            print(f"  {name:20s} - seen {count:2d}x, last: {last_seen}")
    
    def remove_speaker(self, name: str):
        """Remove a speaker from the database."""
        if name in self.speakers:
            del self.speakers[name]
            self.save_database()
            print(f"Removed speaker: {name}")
        else:
            print(f"Speaker not found: {name}")
    
    def export_embeddings(self, output_path: Path = None):
        """Export all embeddings to numpy format for analysis."""
        if output_path is None:
            output_path = self.database_path.with_suffix('.npz')
        
        embeddings_dict = {
            name: data['embedding'] 
            for name, data in self.speakers.items()
        }
        
        np.savez(output_path, **embeddings_dict)
        print(f"Exported embeddings to: {output_path}")


if __name__ == "__main__":
    # Simple CLI for managing speaker database
    import sys
    
    db = SpeakerDatabase()
    
    if len(sys.argv) == 1:
        db.list_speakers()
    elif sys.argv[1] == "list":
        db.list_speakers()
    elif sys.argv[1] == "remove" and len(sys.argv) == 3:
        db.remove_speaker(sys.argv[2])
    elif sys.argv[1] == "export":
        db.export_embeddings()
    else:
        print("Usage:")
        print("  python speaker_database.py           # list speakers")
        print("  python speaker_database.py list      # list speakers")
        print("  python speaker_database.py remove NAME")
        print("  python speaker_database.py export")
