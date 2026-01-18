#!/bin/bash
# Setup script for bamfscribe

set -e

echo "Setting up bamfscribe..."
echo

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "Warning: Python 3.12+ recommended. Current version: $PYTHON_VERSION"
    echo "Consider upgrading for best performance."
    echo
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "✓ Installation complete!"
echo

# Run installation test
echo "Running installation test..."
echo
python test_installation.py

echo
echo "Next steps:"
echo "1. Set up Hugging Face token for speaker diarization:"
echo "   - Accept terms at these URLs:"
echo "     * https://huggingface.co/pyannote/speaker-diarization-community-1 (primary - fastest)"
echo "     * https://huggingface.co/pyannote/segmentation-3.0"
echo "   - Get your token from: https://huggingface.co/settings/tokens"
echo "   - Run: export HF_TOKEN='your_token_here'"
echo "   - Add to ~/.zshrc to make permanent"
echo
echo "2. Grant Full Disk Access (if needed):"
echo "   - System Settings → Privacy & Security → Full Disk Access"
echo "   - Add Terminal or Python"
echo
echo "3. (Optional) Install Cursor CLI for automatic summaries:"
echo "   - Install: curl https://cursor.com/install -fsS | bash"
echo "   - Get API key from: https://cursor.com/dashboard?tab=cloud-agents"
echo "   - Run: export CURSOR_API_KEY='your_api_key_here'"
echo "   - Add to ~/.zshrc to make permanent"
echo "   - Verify: agent --version"
echo
echo "4. (Optional) Set your preferences:"
echo "   cp config.example.py config.py"
echo "   # Edit config.py with your workspace path and preferences"
echo
echo "5. Run the tool:"
echo "   python bamfscribe.py"
echo
