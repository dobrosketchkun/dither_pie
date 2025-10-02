"""
High-performance video processing module with multiprocessing support.
Optimized to prevent freezing and handle large videos efficiently.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from PIL import Image

from dithering_lib import ImageDitherer


class VideoProcessor:
    """
    Handles video processing with intelligent frame handling and parallel processing.
    """
    
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 progress_callback: Optional[Callable[[float, str], None]] = None):
        """
        Initialize video processor.
        
        Args:
            num_workers: Number of parallel workers. Defaults to CPU count - 1.
            progress_callback: Function to call with (progress_fraction, status_message).
        """
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.progress_callback = progress_callback
        
    def _report_progress(self, fraction: float, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(fraction, message)
            
    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video metadata using ffprobe.
        
        Returns:
            dict with keys: fps, width, height, duration, frame_count
        """
        try:
            # Get FPS
            fps_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,width,height,duration,nb_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ]
            result = subprocess.run(fps_cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            # Parse FPS
            fps_str = lines[0] if len(lines) > 0 else "30/1"
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str) if fps_str else 30.0
                
            # Parse dimensions
            width = int(lines[1]) if len(lines) > 1 else 1920
            height = int(lines[2]) if len(lines) > 2 else 1080
            
            # Parse duration and frame count
            duration = float(lines[3]) if len(lines) > 3 and lines[3] != 'N/A' else None
            frame_count = int(lines[4]) if len(lines) > 4 and lines[4] != 'N/A' else None
            
            # Estimate frame count if not available
            if frame_count is None and duration is not None:
                frame_count = int(duration * fps)
                
            return {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration,
                'frame_count': frame_count
            }
        except Exception as e:
            print(f"Warning: Could not get video info: {e}", file=sys.stderr)
            return {'fps': 30.0, 'width': 1920, 'height': 1080, 'duration': None, 'frame_count': None}
    
    def process_video_streaming(self,
                                input_path: str,
                                output_path: str,
                                ditherer: ImageDitherer,
                                pixelize_func: Optional[Callable[[Image.Image], Image.Image]] = None,
                                batch_size: int = 30) -> bool:
        """
        Process video using streaming approach with batched multiprocessing.
        Much faster than frame-by-frame extraction.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            ditherer: ImageDitherer instance
            pixelize_func: Optional function to pixelize frames before dithering
            batch_size: Number of frames to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            video_info = self.get_video_info(input_path)
            fps = video_info['fps']
            total_frames = video_info.get('frame_count', 0)
            
            self._report_progress(0.0, "Initializing video processing...")
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                
                # Extract frames in batches using ffmpeg
                self._report_progress(0.05, "Extracting frames...")
                frame_pattern = str(tmp_dir_path / "frame_%05d.png")
                
                extract_cmd = [
                    "ffmpeg", "-i", input_path,
                    "-qscale:v", "2",  # High quality
                    frame_pattern
                ]
                
                subprocess.run(extract_cmd, 
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             check=True)
                
                # Get list of extracted frames
                frame_files = sorted(tmp_dir_path.glob("frame_*.png"))
                total_frames = len(frame_files)
                
                if total_frames == 0:
                    raise ValueError("No frames extracted from video")
                
                self._report_progress(0.1, f"Processing {total_frames} frames...")
                
                # Process frames in batches using multiprocessing
                processed_count = 0
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_files = frame_files[batch_start:batch_end]
                    
                    # Create processing function for this batch
                    process_func = partial(
                        _process_single_frame,
                        ditherer=ditherer,
                        pixelize_func=pixelize_func
                    )
                    
                    # Process batch in parallel
                    with Pool(processes=self.num_workers) as pool:
                        pool.map(process_func, batch_files)
                    
                    processed_count += len(batch_files)
                    progress = 0.1 + 0.8 * (processed_count / total_frames)
                    self._report_progress(progress, 
                                        f"Processed {processed_count}/{total_frames} frames")
                
                # Re-encode video
                self._report_progress(0.9, "Encoding final video...")
                
                encode_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", f"{fps:.5f}",
                    "-i", frame_pattern,
                    "-i", input_path,
                    "-map", "0:v",
                    "-map", "1:a?",
                    "-map", "1:s?",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-c:s", "copy",
                    output_path
                ]
                
                subprocess.run(encode_cmd,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             check=True)
                
                self._report_progress(1.0, "Video processing complete!")
                return True
                
        except Exception as e:
            self._report_progress(1.0, f"Error: {str(e)}")
            print(f"Video processing error: {e}", file=sys.stderr)
            return False


def _process_single_frame(frame_path: Path,
                          ditherer: ImageDitherer,
                          pixelize_func: Optional[Callable[[Image.Image], Image.Image]] = None):
    """
    Process a single frame. Used by multiprocessing pool.
    Must be a top-level function for pickling.
    """
    try:
        img = Image.open(frame_path).convert('RGB')
        
        # Pixelize if function provided
        if pixelize_func is not None:
            img = pixelize_func(img)
        
        # Apply dithering
        dithered = ditherer.apply_dithering(img)
        
        # Save back to same path
        dithered.save(frame_path)
        
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}", file=sys.stderr)


class NeuralPixelizer:
    """
    Wrapper for neural pixelization model with caching to avoid reloading.
    This prevents the major performance bottleneck in the original code.
    """
    
    _instance = None
    _model = None
    _device = None
    
    def __new__(cls, device: Optional[str] = None):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the neural model (only once due to singleton)."""
        if self._model is None:
            import torch
            from models.pixelization import Model
            
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = Model(device=self._device)
            self._model.load()
            print(f"Neural model loaded on {self._device}")
    
    def pixelize(self, image: Image.Image, max_size: int) -> Image.Image:
        """
        Pixelize an image using the neural model.
        
        Args:
            image: Input PIL Image
            max_size: Target size for smaller dimension
            
        Returns:
            Pixelized PIL Image
        """
        from models.pixelization import resize_image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_input = os.path.join(tmpdir, "temp_in.png")
            temp_output = os.path.join(tmpdir, "temp_out.png")
            temp_resized = os.path.join(tmpdir, "temp_resized.png")
            
            # Save input
            image.save(temp_input)
            
            # Resize for neural processing
            resize_image(temp_input, temp_resized, max_size * 4)
            
            # Pixelize
            self._model.pixelize(temp_resized, temp_output)
            
            # Load result
            result = Image.open(temp_output).convert('RGB')
            
            # Compute even dimensions
            orig_w, orig_h = result.size
            target_w, target_h = self._compute_even_dimensions(orig_w, orig_h, max_size)
            result = result.resize((target_w, target_h), Image.Resampling.NEAREST)
            
            return result
    
    @staticmethod
    def _compute_even_dimensions(orig_w: int, orig_h: int, max_size: int) -> Tuple[int, int]:
        """Compute even dimensions for video codec compatibility."""
        if orig_w >= orig_h:
            target_h = max_size if max_size % 2 == 0 else max_size - 1
            target_w = int(round((orig_w / orig_h) * target_h))
            if target_w % 2 != 0:
                target_w += 1
        else:
            target_w = max_size if max_size % 2 == 0 else max_size - 1
            target_h = int(round((orig_h / orig_w) * target_w))
            if target_h % 2 != 0:
                target_h += 1
        return target_w, target_h


def pixelize_regular(image: Image.Image, max_size: int) -> Image.Image:
    """
    Regular pixelization using nearest-neighbor interpolation.
    
    Args:
        image: Input PIL Image
        max_size: Target size for smaller dimension
        
    Returns:
        Pixelized PIL Image with even dimensions
    """
    orig_w, orig_h = image.size
    target_w, target_h = NeuralPixelizer._compute_even_dimensions(orig_w, orig_h, max_size)
    resized = image.resize((target_w, target_h), Image.Resampling.NEAREST)
    return resized.convert('RGB')

