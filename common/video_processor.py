"""
Unified video processor for YOLO detection systems.
"""
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import torch

if TYPE_CHECKING:
    from detectors.base import BaseDetector


class VideoProcessor:
    """
    Unified video processing class with memory optimization.
    Supports streaming write and GPU memory management.
    """

    def __init__(self, detector: "BaseDetector", output_dir: str):
        """
        Initialize video processor.

        Args:
            detector: Detector instance implementing detect() method
            output_dir: Output directory for processed videos
        """
        self.detector = detector
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_video(
        self,
        video_path: str,
        output_name: Optional[str] = None,
        show_progress: bool = True,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
        progress_interval: int = 30,
        memory_cleanup_interval: int = 50
    ) -> int:
        """
        Process video with object detection.

        Args:
            video_path: Input video path
            output_name: Output filename (auto-generated if None)
            show_progress: Whether to show progress
            max_frames: Maximum frames to process (None for all)
            skip_frames: Number of frames to skip (0 = process all)
            progress_interval: Frames interval for progress display
            memory_cleanup_interval: Frames interval for GPU memory cleanup

        Returns:
            Number of processed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate output FPS
        output_fps = fps // (skip_frames + 1) if skip_frames > 0 else fps

        # Limit total frames if max_frames specified
        if max_frames:
            total_frames = min(total_frames, max_frames * (skip_frames + 1))

        self._print_video_info(video_path, width, height, fps, output_fps, total_frames)

        # Setup output paths
        if output_name is None:
            input_name = Path(video_path).stem
            output_name = f"{input_name}_detected.mp4"
        output_path = f"{self.output_dir}/{output_name}"

        # Use temp file for faster writing
        temp_path = "/tmp/temp_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Frame skipping
                if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                    continue

                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break

                # Run detection
                _, annotated = self.detector.detect(frame)
                out.write(annotated)
                processed_count += 1

                # Show progress
                if show_progress and processed_count % progress_interval == 0:
                    self._print_progress(
                        processed_count, total_frames, skip_frames, start_time
                    )

                # GPU memory cleanup
                if processed_count % memory_cleanup_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            cap.release()
            out.release()

            # Copy to output directory
            if os.path.exists(temp_path):
                print(f"\nCopying to: {output_path}")
                shutil.copy2(temp_path, output_path)
                os.remove(temp_path)

        # Print completion stats
        total_time = time.time() - start_time
        self._print_completion_stats(processed_count, total_time, output_path)

        return processed_count

    def _print_video_info(
        self,
        video_path: str,
        width: int,
        height: int,
        fps: int,
        output_fps: int,
        total_frames: int
    ) -> None:
        """Print video information."""
        print(f"Input: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps} -> {output_fps} (output)")
        print(f"  Total frames: {total_frames}")

    def _print_progress(
        self,
        processed_count: int,
        total_frames: int,
        skip_frames: int,
        start_time: float
    ) -> None:
        """Print processing progress."""
        elapsed = time.time() - start_time
        fps_actual = processed_count / elapsed if elapsed > 0 else 0
        remaining = (total_frames // (skip_frames + 1) - processed_count)
        eta = remaining / fps_actual if fps_actual > 0 else 0
        print(f"Progress: {processed_count} frames processed")
        print(f"Speed: {fps_actual:.1f} fps | ETA: {eta:.0f}s")

    def _print_completion_stats(
        self,
        processed_count: int,
        total_time: float,
        output_path: str
    ) -> None:
        """Print completion statistics."""
        avg_fps = processed_count / total_time if total_time > 0 else 0
        print(f"\nCompleted: {processed_count} frames in {total_time:.1f}s")
        print(f"Average speed: {avg_fps:.1f} fps")
        print(f"Output saved: {output_path}")
