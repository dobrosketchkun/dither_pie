"""
High-performance video processing module with multiprocessing support.
Optimized to prevent freezing and handle large videos efficiently.
"""

import os
import sys
import subprocess
import tempfile
import shutil
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
            num_workers: Number of parallel workers. Defaults to min(4, CPU count - 1) to prevent system freeze.
            progress_callback: Function to call with (progress_fraction, status_message).
        """
        # Limit workers to prevent system freeze - use at most 4 workers
        if num_workers is None:
            num_workers = min(4, max(1, cpu_count() - 1))
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        
    def _report_progress(self, fraction: float, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(fraction, message)
    
    def _fix_failed_frames(self, failed_frames: list, all_frames: list):
        """
        Fix failed frames by copying from the nearest successfully processed frame.
        This ensures video maintains full length without gaps.
        
        Args:
            failed_frames: List of Path objects for frames that failed
            all_frames: List of all frame Path objects in order
        """
        # Create set for fast lookup
        failed_set = set(failed_frames)
        
        for failed_frame in failed_frames:
            # Find the index of this frame
            try:
                idx = all_frames.index(failed_frame)
            except ValueError:
                print(f"Could not find index for {failed_frame.name}", file=sys.stderr)
                continue
            
            # Try to find nearest successful frame
            # First try previous frames (going backwards)
            source_frame = None
            for i in range(idx - 1, -1, -1):
                if all_frames[i] not in failed_set and all_frames[i].exists():
                    source_frame = all_frames[i]
                    break
            
            # If no previous frame, try next frames (going forward)
            if source_frame is None:
                for i in range(idx + 1, len(all_frames)):
                    if all_frames[i] not in failed_set and all_frames[i].exists():
                        source_frame = all_frames[i]
                        break
            
            # Copy the source frame
            if source_frame:
                try:
                    shutil.copy2(source_frame, failed_frame)
                    print(f"Fixed {failed_frame.name} by copying from {source_frame.name}", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to copy frame {source_frame.name} to {failed_frame.name}: {e}", file=sys.stderr)
            else:
                print(f"ERROR: Could not find any successful frame to copy for {failed_frame.name}", file=sys.stderr)
            
    def get_video_info(self, video_path: str) -> dict:
        """
        Extract video metadata using ffprobe.
        
        Returns:
            dict with keys: fps, width, height, duration, frame_count
        """
        try:
            # Get FPS separately for more reliable parsing
            fps_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ]
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True, check=True)
            fps_str = fps_result.stdout.strip()
            
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str) if fps_str else 30.0
            
            # Get dimensions separately
            dims_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ]
            dims_result = subprocess.run(dims_cmd, capture_output=True, text=True, check=True)
            dims_lines = dims_result.stdout.strip().split('\n')
            width = int(dims_lines[0]) if len(dims_lines) > 0 else 1920
            height = int(dims_lines[1]) if len(dims_lines) > 1 else 1080
            
            # Get duration and frame count
            info_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration,nb_frames",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path
            ]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            info_lines = info_result.stdout.strip().split('\n')
            
            duration = None
            frame_count = None
            for line in info_lines:
                if line and line != 'N/A':
                    try:
                        val = float(line)
                        if val > 100:  # Likely frame count
                            frame_count = int(val)
                        else:  # Likely duration
                            duration = val
                    except:
                        pass
            
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
                                batch_size: int = 15) -> bool:
        """
        Process video using streaming approach with batched multiprocessing.
        Much faster than frame-by-frame extraction.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            ditherer: ImageDitherer instance
            pixelize_func: Optional function to pixelize frames before dithering
            batch_size: Number of frames to process in each batch (reduced to 15 to prevent freeze)
            
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
                
                # Unpack pixelize_func (now a tuple like ("neural", 64) or None)
                pixelize_method = None
                max_size = 64
                if pixelize_func is not None:
                    pixelize_method, max_size = pixelize_func
                
                # Determine processing strategy based on pixelization method
                # Neural pixelization: process in main thread to reuse loaded model
                # Regular pixelization: use multiprocessing for speed
                
                processed_count = 0
                
                if pixelize_method == "neural":
                    # Process sequentially in main thread to avoid reloading model
                    self._report_progress(0.1, f"Processing {total_frames} frames (neural mode: slower but higher quality)...")
                    
                    # Load model once in main process
                    pixelizer = NeuralPixelizer()
                    
                    failed_frames = []
                    for frame_file in frame_files:
                        success = False
                        last_error = None
                        
                        # Try up to 3 times
                        for attempt in range(3):
                            try:
                                img = Image.open(frame_file).convert('RGB')
                                # Pixelize using shared model instance
                                img = pixelizer.pixelize(img, max_size)
                                # Apply dithering
                                dithered = ditherer.apply_dithering(img)
                                # Save back
                                dithered.save(frame_file)
                                
                                # Verify the frame was saved correctly
                                if not frame_file.exists() or frame_file.stat().st_size == 0:
                                    raise ValueError(f"Frame {frame_file} not saved properly")
                                
                                success = True
                                break
                            except Exception as e:
                                last_error = e
                                if attempt < 2:  # Don't print on last attempt
                                    print(f"Retry {attempt+1}/3 for frame {frame_file.name}: {e}", file=sys.stderr)
                        
                        if not success:
                            print(f"Error processing frame {frame_file} after 3 attempts: {last_error}", file=sys.stderr)
                            failed_frames.append(frame_file)
                        
                        processed_count += 1
                        if processed_count % 5 == 0:  # Report every 5 frames
                            progress = 0.1 + 0.8 * (processed_count / total_frames)
                            self._report_progress(progress, 
                                                f"Processed {processed_count}/{total_frames} frames")
                    
                    # Handle failed frames by copying from nearest successful frame
                    if failed_frames:
                        print(f"Fixing {len(failed_frames)} failed frames by copying from nearest frames...", file=sys.stderr)
                        self._fix_failed_frames(failed_frames, frame_files)
                else:
                    # Use multiprocessing for regular pixelization
                    failed_frames = []
                    for batch_start in range(0, total_frames, batch_size):
                        batch_end = min(batch_start + batch_size, total_frames)
                        batch_files = frame_files[batch_start:batch_end]
                        
                        # Create processing function for this batch
                        process_func = partial(
                            _process_single_frame,
                            ditherer=ditherer,
                            pixelize_method=pixelize_method,
                            max_size=max_size
                        )
                        
                        # Process batch in parallel
                        with Pool(processes=self.num_workers) as pool:
                            results = pool.map(process_func, batch_files)
                        
                        # Check for failures and retry them sequentially
                        for i, success in enumerate(results):
                            if not success:
                                frame_path = batch_files[i]
                                print(f"Retrying failed frame {frame_path.name}...", file=sys.stderr)
                                # Retry up to 2 more times
                                retry_success = False
                                for retry in range(2):
                                    if _process_single_frame(frame_path, ditherer, pixelize_method, max_size):
                                        retry_success = True
                                        break
                                if not retry_success:
                                    failed_frames.append(frame_path)
                        
                        processed_count += len(batch_files)
                        progress = 0.1 + 0.8 * (processed_count / total_frames)
                        self._report_progress(progress, 
                                            f"Processed {processed_count}/{total_frames} frames")
                    
                    # Handle failed frames by copying from nearest successful frame
                    if failed_frames:
                        print(f"Fixing {len(failed_frames)} failed frames by copying from nearest frames...", file=sys.stderr)
                        self._fix_failed_frames(failed_frames, frame_files)
                
                # Re-encode video
                self._report_progress(0.9, "Encoding final video...")
                
                # Verify all frames exist - they should now after fix_failed_frames
                existing_frames = sorted(tmp_dir_path.glob("frame_*.png"))
                if len(existing_frames) != total_frames:
                    print(f"WARNING: Expected {total_frames} frames but found {len(existing_frames)}", 
                          file=sys.stderr)
                
                print(f"Encoding {total_frames} frames at {fps:.3f} fps", file=sys.stderr)
                
                # CRITICAL FIX: Use -vframes to specify exact number of frames
                # This prevents ffmpeg from duplicating the last frame to match audio duration
                encode_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", f"{fps:.5f}",
                    "-i", frame_pattern,
                    "-i", input_path,
                    "-map", "0:v:0",  # Take video from first input (our frames)
                    "-map", "1:a?",   # Take audio from second input (original video) if it exists
                    "-map", "1:s?",   # Take subtitles from second input if they exist
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-vframes", str(total_frames),  # Output option: encode exactly this many frames
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
                          pixelize_method: Optional[str] = None,
                          max_size: int = 64) -> bool:
    """
    Process a single frame. Used by multiprocessing pool.
    Must be a top-level function for pickling.
    
    Args:
        frame_path: Path to the frame image
        ditherer: ImageDitherer instance for dithering
        pixelize_method: "neural" or "regular" or None for no pixelization
        max_size: Maximum size for pixelization
    
    Returns:
        True if successful, False if failed
    """
    try:
        img = Image.open(frame_path).convert('RGB')
        
        # Pixelize if method provided
        if pixelize_method == "neural":
            # Use neural pixelization
            pixelizer = NeuralPixelizer()
            img = pixelizer.pixelize(img, max_size)
        elif pixelize_method == "regular":
            # Use regular pixelization
            img = pixelize_regular(img, max_size)
        
        # Apply dithering
        dithered = ditherer.apply_dithering(img)
        
        # Save back to same path
        dithered.save(frame_path)
        
        # Verify the frame was saved correctly
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            raise ValueError(f"Frame {frame_path} not saved properly")
        
        return True
        
    except Exception as e:
        print(f"Error processing frame {frame_path}: {e}", file=sys.stderr)
        return False


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

