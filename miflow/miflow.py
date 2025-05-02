"""
MiFlow: Depth-Enhanced Video Stabilization

This module implements the MiFlow video stabilization algorithm based on the paper
"MiFlow: Combining MiDaS Depth and Optical Flow for Enhanced Video Stabilization"
"""

import cv2
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image


class MiDaSModel(nn.Module):
    """MiDaS v2.1 Small model for monocular depth estimation"""

    def __init__(self, model_weights_path=None):
        super().__init__()
        # Load MiDaS model
        try:
            print("Loading MiDaS model...")
            # Use the specific model version that's more compatible
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            if model_weights_path and os.path.exists(model_weights_path):
                state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            print("Falling back to CPU execution and downloading the model")
            try:
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
            except Exception as e2:
                print(f"Failed to load model even with fallback: {e2}")
                raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

        # Define image transforms
        self.transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, image):
        """Predict depth from image"""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image_tensor)

            # Get image dimensions for proper resizing
            if isinstance(image, Image.Image):
                # PIL Image
                width, height = image.size
                size = (height, width)
            else:
                # NumPy array
                size = image.shape[:2]

            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=size,
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        return prediction


class MiFlow:
    """MiFlow video stabilizer using optical flow and depth estimation"""

    def __init__(self, use_depth=True, smooth_radius=30, smooth_strength=25,
                 crop_ratio=0.9, midas_weights_path=None, verbose=False):
        """
        Initialize MiFlow video stabilizer

        Args:
            use_depth (bool): Whether to use depth information for stabilization
            smooth_radius (int): Radius for Gaussian smoothing window
            smooth_strength (int): Strength of smoothing (σ of Gaussian)
            crop_ratio (float): Crop ratio to remove borders (0-1)
            midas_weights_path (str): Path to MiDaS model weights
            verbose (bool): Print additional information
        """
        self.use_depth = use_depth
        self.smooth_radius = smooth_radius
        self.smooth_strength = smooth_strength
        self.crop_ratio = crop_ratio
        self.verbose = verbose

        # Initialize trajectories
        self.trajectories = {'x': [], 'y': []}
        self.smoothed_trajectories = {'x': [], 'y': []}
        self.prev_gray = None
        self.frame_shape = None

        # Initialize MiDaS model if depth is used
        if self.use_depth:
            if self.verbose:
                print("Initializing MiDaS depth model...")
            self.depth_model = MiDaSModel(midas_weights_path)

        # Farneback optical flow parameters
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

    def _get_depth_map(self, frame):
        """Get depth map from MiDaS model"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for the transform pipeline
        pil_image = Image.fromarray(frame_rgb)

        depth = self.depth_model(pil_image)

        # Normalize depth for visualization and computation
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        return depth_norm

    def _compute_optical_flow(self, frame):
        """Compute optical flow between consecutive frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.farneback_params
        )

        self.prev_gray = gray
        return flow

    def _update_trajectory(self, flow, depth_map=None):
        """Update trajectory based on optical flow and depth map"""
        if flow is None:
            self.trajectories['x'].append(0)
            self.trajectories['y'].append(0)
            return

        h, w = flow.shape[:2]

        if depth_map is not None and self.use_depth:
            # Weight flow vectors by depth (far pixels get higher weight)
            flow_weights = depth_map.reshape(h, w, 1)
            weighted_flow = flow * flow_weights

            # Compute average displacement weighted by depth
            total_weight = np.sum(flow_weights)
            if total_weight > 0:
                dx = np.sum(weighted_flow[:, :, 0]) / total_weight
                dy = np.sum(weighted_flow[:, :, 1]) / total_weight
            else:
                dx, dy = 0, 0
        else:
            # Simple average of flow vectors
            dx = np.mean(flow[:, :, 0])
            dy = np.mean(flow[:, :, 1])

        # Append to trajectory
        self.trajectories['x'].append(dx)
        self.trajectories['y'].append(dy)

    def _smooth_trajectory(self):
        """Apply Gaussian smoothing to trajectories"""
        # Create cumulative trajectories from individual motions
        cum_x = np.cumsum(self.trajectories['x'])
        cum_y = np.cumsum(self.trajectories['y'])

        # Define Gaussian kernel
        radius = self.smooth_radius
        sigma = self.smooth_strength

        # Ensure radius is odd
        if radius % 2 == 0:
            radius += 1

        # Create kernel
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel_1d = kernel.flatten()

        # Apply convolution for smoothing
        smoothed_x = cv2.filter2D(cum_x.reshape(-1, 1), -1, kernel_1d).flatten()
        smoothed_y = cv2.filter2D(cum_y.reshape(-1, 1), -1, kernel_1d).flatten()

        # Handle edges
        half_radius = radius // 2
        smoothed_x[:half_radius] = cum_x[:half_radius]
        smoothed_y[:half_radius] = cum_y[:half_radius]
        smoothed_x[-half_radius:] = cum_x[-half_radius:]
        smoothed_y[-half_radius:] = cum_y[-half_radius:]

        return smoothed_x, smoothed_y

    def _compute_transforms(self):
        """Compute transformation matrices based on smoothed trajectories"""
        smoothed_x, smoothed_y = self._smooth_trajectory()

        self.smoothed_trajectories['x'] = smoothed_x
        self.smoothed_trajectories['y'] = smoothed_y

        # Calculate difference between original and smoothed trajectories
        transforms = []
        cum_x = np.cumsum(self.trajectories['x'])
        cum_y = np.cumsum(self.trajectories['y'])

        for i in range(len(cum_x)):
            # Transformation to stabilize the frame
            dx = smoothed_x[i] - cum_x[i]
            dy = smoothed_y[i] - cum_y[i]

            transform = np.array([[1, 0, dx],
                                  [0, 1, dy]])
            transforms.append(transform)

        return transforms

    def _apply_transform(self, frame, transform):
        """Apply affine transformation to frame"""
        h, w = frame.shape[:2]

        # Apply crop margin to prevent black borders
        if self.crop_ratio < 1.0:
            crop_h = int(h * self.crop_ratio)
            crop_w = int(w * self.crop_ratio)

            # Calculate cropping parameters
            h_margin = (h - crop_h) // 2
            w_margin = (w - crop_w) // 2

            # Adjust transform to account for cropping
            transform[0, 2] += w_margin
            transform[1, 2] += h_margin

            # Apply transform
            stabilized = cv2.warpAffine(frame, transform, (w, h))

            # Crop the image
            stabilized = stabilized[h_margin:h - h_margin, w_margin:w - w_margin]

            # Resize back to original size
            stabilized = cv2.resize(stabilized, (w, h))
        else:
            # No cropping, just apply transform
            stabilized = cv2.warpAffine(frame, transform, (w, h))

        return stabilized

    def process_video(self, input_path, output_path=None, preview=False):
        """
        Process video with MiFlow stabilization

        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video (None for preview only)
            preview (bool): Show preview during processing

        Returns:
            output_path (str): Path to output video if successful
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_shape = (height, width)

        # Initialize output video
        out = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if self.verbose:
            print(f"Processing video: {input_path}")
            print(f"Resolution: {width}x{height}, FPS: {fps}")
            print(f"Using depth: {self.use_depth}")

        # First pass: collect trajectories
        self.trajectories = {'x': [], 'y': []}
        self.prev_gray = None

        # Use tqdm for progress bar
        if self.verbose:
            print("Pass 1: Analyzing motion...")

        # Frame buffers for two-pass processing
        frames_buffer = []

        for _ in tqdm(range(frame_count), disable=not self.verbose):
            ret, frame = cap.read()
            if not ret:
                break

            # Store frame for second pass
            frames_buffer.append(frame.copy())

            # Calculate optical flow
            flow = self._compute_optical_flow(frame)

            # Get depth map if enabled
            depth_map = None
            if self.use_depth:
                depth_map = self._get_depth_map(frame)

            # Update trajectory
            self._update_trajectory(flow, depth_map)

        # Compute transforms based on trajectories
        transforms = self._compute_transforms()

        # Second pass: apply stabilization
        if self.verbose:
            print("Pass 2: Applying stabilization...")

        for i, frame in enumerate(tqdm(frames_buffer, disable=not self.verbose)):
            if i >= len(transforms):
                break

            # Apply stabilization transform
            stabilized = self._apply_transform(frame, transforms[i])

            # Write to output file
            if out is not None:
                out.write(stabilized)

            # Show preview
            if preview:
                # Display original and stabilized frames side by side
                if width > 640:
                    scale = 640 / width
                    display_width = int(width * scale)
                    display_height = int(height * scale)

                    frame_resized = cv2.resize(frame, (display_width, display_height))
                    stabilized_resized = cv2.resize(stabilized, (display_width, display_height))

                    # Create comparison view
                    comparison = np.hstack((frame_resized, stabilized_resized))
                    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, "Stabilized", (display_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                    cv2.imshow("MiFlow Stabilization", comparison)
                else:
                    comparison = np.hstack((frame, stabilized))
                    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(comparison, "Stabilized", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("MiFlow Stabilization", comparison)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        if out is not None:
            out.release()

        if preview:
            cv2.destroyAllWindows()

        if self.verbose and output_path is not None:
            print(f"Stabilized video saved to: {output_path}")

        return output_path

    def plot_trajectories(self, output_path=None):
        """
        Plot original and smoothed trajectories

        Args:
            output_path (str): Path to save the trajectory plot image
        """
        try:
            import matplotlib.pyplot as plt

            cum_x = np.cumsum(self.trajectories['x'])
            cum_y = np.cumsum(self.trajectories['y'])

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.plot(cum_x, label='Original')
            plt.plot(self.smoothed_trajectories['x'], label='Smoothed')
            plt.title('Horizontal Trajectory')
            plt.ylabel('Displacement')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(cum_y, label='Original')
            plt.plot(self.smoothed_trajectories['y'], label='Smoothed')
            plt.title('Vertical Trajectory')
            plt.xlabel('Frame')
            plt.ylabel('Displacement')
            plt.legend()

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path)
                if self.verbose:
                    print(f"Trajectory plot saved to: {output_path}")

            plt.show()

        except ImportError:
            print("Matplotlib is required for plotting. Install with: pip install matplotlib")
