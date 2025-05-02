"""
Setup script for MiFlow package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="miflow",
    version="0.1.0",
    author="MiFlow Team",
    author_email="bhavisha2705@gmail.com",
    description="Depth-Enhanced Video Stabilization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bhavisha-06/MiFlow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "tqdm>=4.50.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "plot": ["matplotlib>=3.3.0"],
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
        ],
    },
    entry_points={
        "console_scripts": [
            "miflow=miflow.cli:main",
        ],
    },
)
