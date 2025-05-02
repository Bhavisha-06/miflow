#!/usr/bin/env python
"""
MiFlow Command Line Interface

This module provides a command-line interface for the MiFlow video stabilization tool.
"""

import argparse
import os
import sys
import time
from miflow import MiFlow


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MiFlow - Depth-Enhanced Video Stabilization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", help="Path to input video file")
    
    parser.add_argument("-o", "--output", 
                        help="Path to output video file (default: input_stabilized.mp4)")
    
    parser.add_argument("--no-depth", action="store_true", 
                        help="Disable depth information (faster but less accurate)")
    
    parser.add_argument("--smooth-radius", type=int, default=30,
                        help="Radius for Gaussian smoothing window")
    
    parser.add_argument("--smooth-strength", type=int, default=25,
                        help="Strength of smoothing (σ of Gaussian)")
    
    parser.add_argument("--crop-ratio", type=float, default=0.95,
                        help="Crop ratio to remove borders (0-1)")
    
    parser.add_argument("--midas-weights", 
                        help="Path to custom MiDaS model weights")
    
    parser.add_argument("--preview", action="store_true",
                        help="Show preview during processing")
    
    parser.add_argument("--plot", action="store_true",
                        help="Plot trajectories after processing")
    
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI"""
    args = parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input video file not found at {args.input}")
        return 1

    # Generate output path if not provided
    if not args.output:
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        args.output = os.path.join(os.path.dirname(args.input), f"{name}_stabilized{ext}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Print configuration if not quiet
    if not args.quiet:
        print("MiFlow Video Stabilization")
        print(f"Input video: {args.input}")
        print(f"Output will be saved to: {args.output}")
        print(f"Using depth: {not args.no_depth}")
        print(f"Smooth radius: {args.smooth_radius}")
        print(f"Smooth strength: {args.smooth_strength}")
        print(f"Crop ratio: {args.crop_ratio}")

    try:
        # Initialize MiFlow
        miflow = MiFlow(
            use_depth=not args.no_depth,
            smooth_radius=args.smooth_radius,
            smooth_strength=args.smooth_strength,
            crop_ratio=args.crop_ratio,
            midas_weights_path=args.midas_weights,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error initializing MiFlow: {e}")
        return 1

    # Process video
    try:
        start_time = time.time()
        miflow.process_video(args.input, args.output, preview=args.preview)
        elapsed_time = time.time() - start_time
        
        if not args.quiet:
            print(f"Processing completed in {elapsed_time:.2f} seconds")

        # Plot trajectories if requested
        if args.plot:
            plot_output_path = os.path.join(os.path.dirname(args.output), 
                                           f"{os.path.splitext(os.path.basename(args.output))[0]}_trajectories.png")
            miflow.plot_trajectories(plot_output_path)
            
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
