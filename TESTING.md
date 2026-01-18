# Testing Notes

## Parakeet-MLX API

The parakeet-mlx implementation in this tool is based on the expected API pattern for MLX-based transcription models. If you encounter issues, the API may have changed.

### Common API patterns to try:

1. **Current implementation:**
```python
import parakeet_mlx as parakeet
result = parakeet.transcribe(audio_path, model="parakeet-medium-en", word_timestamps=True)
```

2. **Alternative pattern 1:**
```python
from parakeet_mlx import Parakeet
model = Parakeet("parakeet-medium-en")
result = model.transcribe(audio_path)
```

3. **Alternative pattern 2:**
```python
import parakeet_mlx
result = parakeet_mlx.transcribe(audio_path, model_size="medium")
```

### If parakeet-mlx API differs:

1. Check the official documentation: https://github.com/ml-explore/mlx-examples
2. Update the `_transcribe_with_parakeet()` method in `bamfscribe.py`
3. Open an issue or PR with the correct API

### Testing checklist:

- [ ] Parakeet-MLX installs correctly
- [ ] Can load parakeet models
- [ ] Transcription produces segments with timestamps
- [ ] Speaker diarization works
- [ ] SRT output is correctly formatted
- [ ] Cursor integration triggers
- [ ] Summary generation works

### Test with a short recording:

```bash
# Test with small model first (fastest)
python bamfscribe.py --backend parakeet --model small --no-summary

# Compare with mlx-whisper
python bamfscribe.py --backend mlx-whisper --model small --no-summary
```

## Known Issues

1. **Parakeet-MLX availability**: Check if parakeet-mlx is available via pip. If not, the tool defaults to mlx-whisper.
2. **API changes**: MLX ecosystem packages may have different APIs than expected.
3. **Model names**: Verify correct model names for both backends.

## Reporting Issues

If you find issues with the parakeet-mlx integration, please include:
- parakeet-mlx version
- Full error message
- Output of `pip list | grep parakeet`
