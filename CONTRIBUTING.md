# Contributing to bamfscribe

Thank you for your interest in contributing to bamfscribe!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bamfscribe.git
   cd bamfscribe
   ```
   
   Original repository: https://github.com/bamford/bamfscribe
3. Follow the setup instructions in [README.md](README.md)

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test your changes:
   ```bash
   python test_installation.py
   python bamfscribe.py --latest  # Test with a sample recording
   ```
4. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of your changes"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and modular

## Testing

Before submitting a PR, please ensure:
- The installation test passes: `python test_installation.py`
- Your changes work with both parakeet and mlx-whisper backends
- Speaker recognition still functions correctly
- Documentation is updated if needed

## Reporting Issues

When reporting bugs, please include:
- Your OS and Python version
- Steps to reproduce the issue
- Error messages or logs
- Expected vs actual behavior

## Feature Requests

We welcome feature suggestions! Please open an issue describing:
- The use case
- Why it would be valuable
- How you envision it working

## Questions?

Feel free to open an issue for questions about contributing.
