# MiFlow Installation Guide

This guide provides detailed installation instructions for the MiFlow video stabilization package.

## Prerequisites

MiFlow requires:

- Python 3.7 or newer
- CUDA-capable GPU (recommended but not required)
- FFmpeg (optional, for better video I/O support)

## Installation Methods

### Method 1: Simple Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/Bhavisha-06/miflow.git
cd miflow

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/Bhavisha-06/miflow.git
cd miflow

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Method 3: Using pip

```bash
pip install git+https://github.com/Bhavisha-06/miflow.git
```

### Method 4: Using Docker

```bash
# Clone the repository
git clone https://github.com/Bhavisha-06/miflow.git
cd miflow

# Build the Docker image
docker build -t miflow .

# Run MiFlow with Docker
docker run -v /path/to/videos:/data miflow input.mp4 -o output.mp4
```

Alternatively, use Docker Compose:

```bash
# Create input and output directories
mkdir -p input output

# Place your video in the input directory
cp your_video.mp4 input/

# Run with Docker Compose
docker-compose run miflow input/your_video.mp4 -o output/stabilized.mp4
```

## Platform-Specific Notes

### Windows

- For OpenCV, you might need Microsoft Visual C++ Build Tools
- For PyTorch with CUDA, check the [PyTorch website](https://pytorch.org/) for the correct installation command

### macOS

- OpenCV can be installed via Homebrew: `brew install opencv`
- PyTorch will use CPU by default as CUDA is not supported on macOS

### Linux

- Ensure development libraries are installed: `apt-get install python3-dev libgl1-mesa-glx`
- For CUDA support, install appropriate NVIDIA drivers and CUDA toolkit

## Testing Your Installation

After installation, verify that MiFlow is working correctly:

```bash
# Run the CLI help command
miflow --help

# Try processing a small test video
miflow /path/to/test_video.mp4 -o /path/to/output.mp4 --no-depth
```

## Common Issues and Solutions

### ImportError: libGL.so.1

```bash
apt-get update && apt-get install -y libgl1-mesa-glx
```

### CUDA not found

Make sure you have installed the CUDA toolkit and the correct PyTorch version:

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

### Memory Issues with Large Videos

For large videos, try:

1. Using `--no-depth` option for less memory usage
2. Processing a smaller section of the video first
3. Reducing the video resolution before processing

## Getting Help

If you encounter any issues during installation, please:

1. Check the [GitHub Issues](https://github.com/Bhavisha-06/miflow/issues) for similar problems
2. Open a new issue with details about your setup and the error message
