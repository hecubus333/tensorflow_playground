# TensorFlow Playground

This repository contains a TensorFlow setup optimized for macOS with Apple Silicon, using Python 3.10. It includes a test script to verify the TensorFlow installation and basic functionality.

## Setup Instructions

1. Install Python 3.10 using Homebrew:
```bash
brew install python@3.10
```

2. Install TensorFlow and required dependencies:
```bash
python3.10 -m pip install tensorflow
```

## Test Script

The repository includes `test_tensorflow.py` which verifies:
- TensorFlow installation
- CPU device availability
- Basic matrix multiplication functionality

To run the test:
```bash
python3.10 test_tensorflow.py
```

## Current Configuration

- TensorFlow Version: 2.18.0
- Python Version: 3.10
- Hardware: Apple Silicon
- Computation: CPU-only configuration

## Notes

- This setup uses CPU computation for maximum compatibility
- TensorFlow logging is suppressed for cleaner output
- The test script includes basic matrix operations to verify functionality
