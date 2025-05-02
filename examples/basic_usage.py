#!/usr/bin/env python
"""
Basic usage example for MiFlow video stabilization
"""

import os
import argparse
from miflow import MiFlow


def main():
    """
    Example of using MiFlow API to stabilize a video
    """
    parser = argparse.ArgumentParser(description="MiFlow Basic Example")
    parser.add_argument("input", help="Path to input video file")
    args = parser.parse_args()

    # Input video path
    input_path = args.input

    # Generate output path based on input
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_stabilized{ext}")

    print(f"Input video: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize MiFlow with default parameters
    stabilizer = MiFlow(
        use_depth=True,           # Use depth information
        smooth_radius=30,         # Radius for Gaussian smoothing
        smooth_strength=25,       # Smoothing strength
        crop_ratio=0.95,          # Crop ratio to remove borders
        verbose=True              # Print progress information
    )

    # Process video
    stabilizer.process_video(
        input_path=input_path,
        output_path=output_path,
        preview=True              # Show preview during processing
    )

    # Plot trajectories
    trajectories_path = os.path.join(os.path.dirname(output_path), f"{name}_trajectories.png")
    stabilizer.plot_trajectories(trajectories_path)


if __name__ == "__main__":
    main()
