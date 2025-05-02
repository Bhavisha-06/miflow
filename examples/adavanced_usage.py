#!/usr/bin/env python
"""
Advanced usage example for MiFlow video stabilization
"""

import os
import argparse
import time
from miflow import MiFlow


def main():
    """
    Example of using MiFlow API with custom parameters
    """
    parser = argparse.ArgumentParser(description="MiFlow Advanced Example")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--use-depth", type=bool, default=True, 
                      help="Use depth information (default: True)")
    parser.add_argument("--smooth-radius", type=int, default=30,
                      help="Radius for Gaussian smoothing window (default: 30)")
    parser.add_argument("--smooth-strength", type=int, default=25,
                      help="Strength of smoothing (default: 25)")
    parser.add_argument("--crop-ratio", type=float, default=0.95,
                      help="Crop ratio to remove borders (default: 0.95)")
    args = parser.parse_args()

    # Input video path
    input_path = args.input

    # Generate output path based on input
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_stabilized{ext}")

    print(f"Input video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Parameters:")
    print(f"  - Use depth: {args.use_depth}")
    print(f"  - Smooth radius: {args.smooth_radius}")
    print(f"  - Smooth strength: {args.smooth_strength}")
    print(f"  - Crop ratio: {args.crop_ratio}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize MiFlow with custom parameters
    stabilizer = MiFlow(
        use_depth=args.use_depth,             # Use depth information
        smooth_radius=args.smooth_radius,     # Radius for Gaussian smoothing
        smooth_strength=args.smooth_strength, # Smoothing strength
        crop_ratio=args.crop_ratio,           # Crop ratio to remove borders
        verbose=True                          # Print progress information
    )

    # Process video with performance measurement
    start_time = time.time()
    stabilizer.process_video(
        input_path=input_path,
        output_path=output_path,
        preview=True              # Show preview during processing
    )
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

    # Plot trajectories
    trajectories_path = os.path.join(os.path.dirname(output_path), f"{name}_trajectories.png")
    stabilizer.plot_trajectories(trajectories_path)


if __name__ == "__main__":
    main()
