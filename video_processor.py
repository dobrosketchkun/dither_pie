"""Bounded video decoding, frame processing, encoding, and audio/subtitle muxing."""

from contextlib import ExitStack
from fractions import Fraction
import json
import math
from multiprocessing import Pool, cpu_count
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional, Callable, Tuple

from PIL import Image
from dithering_lib import ImageDitherer, PixelizeMethod
# Compatibility exports for existing callers.
from pixelization import NeuralPixelizer, pixelize_regular

__all__ = ['VideoProcessor', 'NeuralPixelizer', 'pixelize_regular']


class VideoProcessor:
    def __init__(self, num_workers=None, progress_callback=None):
        if num_workers is None:
            num_workers = min(4, max(1, cpu_count() - 1))
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError('num_workers must be a positive integer')
        self.num_workers = num_workers
        self.progress_callback = progress_callback

    def _report_progress(self, fraction, message):
        if self.progress_callback:
            self.progress_callback(fraction, message)

    def get_video_info(self, video_path: str) -> dict:
        """Read named ffprobe fields; never invent dimensions or a frame rate."""
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries',
            'stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:'
            'format=duration:stream_tags=rotate:stream_side_data=rotation',
            '-of', 'json', str(video_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Cannot probe video: {error.stderr.strip()}") from error
        return _parse_video_info(json.loads(result.stdout))

    def process_video_streaming(self, input_path: str, output_path: str,
                                ditherer: Optional[ImageDitherer],
                                pixelize_func: Optional[Tuple[str, int]] = None,
                                batch_size: int = 15,
                                final_resize_multiplier: Optional[int] = None) -> bool:
        """Stream RGB frames in bounded batches through one persistent pool.

        pixelize_func is the legacy (method, max_size) tuple. Neural inference
        stays in the calling process to reuse its model. Variable-rate input is
        resampled to the probed average frame rate; rational rates stay exact.
        Only compressed video and mux output are staged on disk. The destination
        is replaced after success, so errors/cancellation preserve existing files.
        """
        try:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError('batch_size must be a positive integer')
            if final_resize_multiplier is not None and (
                isinstance(final_resize_multiplier, bool) or
                not isinstance(final_resize_multiplier, int) or final_resize_multiplier < 1
            ):
                raise ValueError('final_resize_multiplier must be a positive integer')
            source, destination = Path(input_path).resolve(), Path(output_path).resolve()
            if source == destination:
                raise ValueError('Input and output video paths must differ')
            info = self.get_video_info(str(source))
            size = (info['width'], info['height'])
            rate = info['fps_fraction']
            method, max_size = pixelize_func if pixelize_func is not None else (None, 64)
            if method not in (None, PixelizeMethod.NONE.value, PixelizeMethod.REGULAR.value,
                              PixelizeMethod.NEURAL.value):
                raise ValueError(f'Unknown pixelization method: {method}')
            neural = NeuralPixelizer() if method == PixelizeMethod.NEURAL.value else None
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._report_progress(0.0, 'Starting streaming video processing...')

            with tempfile.TemporaryDirectory(prefix='dither_', dir=destination.parent) as directory:
                stage = Path(directory)
                encoded = stage / 'video.mp4'
                muxed = stage / ('result' + (destination.suffix or '.mp4'))
                with ExitStack() as resources:
                    decode_errors = resources.enter_context(tempfile.TemporaryFile())
                    decode_command = [
                        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                        '-map', '0:v:0', '-an', '-sn', '-dn',
                        '-vf', f'fps={rate}', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1',
                    ]
                    decoder = subprocess.Popen(decode_command, stdin=subprocess.DEVNULL,
                                               stdout=subprocess.PIPE, stderr=decode_errors,
                                               **_process_options())
                    resources.callback(_stop_process, decoder)
                    frame_bytes = size[0] * size[1] * 3
                    first_data = _read_frame(decoder.stdout, frame_bytes)
                    if first_data is None:
                        _check_process(decoder, decode_errors, 'Decoding')
                        raise ValueError('No frames decoded from video')
                    first_image = Image.frombytes('RGB', size, first_data)
                    first = _process_frame_image(first_image, ditherer, method, max_size,
                                                 final_resize_multiplier, neural)
                    output_size = first.size
                    # Processing the first frame also initializes any automatic
                    # palette before the ditherer is sent to worker processes.
                    encode_errors = resources.enter_context(tempfile.TemporaryFile())
                    encode_command = [
                        'ffmpeg', '-nostdin', '-v', 'error', '-y',
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s:v', f'{output_size[0]}x{output_size[1]}', '-framerate', rate,
                        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '18', '-pix_fmt', 'yuv420p', str(encoded),
                    ]
                    encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE,
                                               stdout=subprocess.DEVNULL, stderr=encode_errors,
                                               **_process_options())
                    resources.callback(_stop_process, encoder)
                    # Cap queued pixel payloads as well as the number of frames.
                    # Algorithm workspaces and worker interpreter memory are additional.
                    payload_per_frame = frame_bytes + output_size[0] * output_size[1] * 3
                    limit = min(batch_size, max(1, (64 * 1024 * 1024) // payload_per_frame))
                    pool = None
                    if neural is None and self.num_workers > 1:
                        pool = resources.enter_context(Pool(
                            processes=self.num_workers, initializer=_init_frame_worker,
                            initargs=(size, output_size, ditherer, method, max_size, final_resize_multiplier),
                        ))
                    processed = 1
                    try:
                        encoder.stdin.write(first.tobytes())
                        del first, first_image, first_data
                        for batch in _frame_batches(decoder.stdout, frame_bytes, limit):
                            if pool is not None:
                                results = pool.map(_process_stream_frame, batch)
                            else:
                                results = []
                                for data in batch:
                                    frame = Image.frombytes('RGB', size, data)
                                    result = _process_frame_image(frame, ditherer, method, max_size,
                                                                  final_resize_multiplier, neural)
                                    if result.size != output_size:
                                        raise ValueError('Processed frame dimensions changed')
                                    results.append(result.tobytes())
                            for data in results:
                                encoder.stdin.write(data)
                            processed += len(batch)
                            total = info['frame_count']
                            fraction = min(0.89, 0.1 + 0.79 * processed / total) if total else 0.1
                            self._report_progress(fraction, f'Processed {processed} frames')
                            del results, batch
                        if pool is not None:
                            # Fork-based workers can inherit pipe descriptors.
                            # Reap them before signaling EOF to the encoder.
                            pool.close()
                            pool.join()
                        _check_process(decoder, decode_errors, 'Decoding')
                        encoder.stdin.close()
                        encoder.stdin = None
                        _check_process(encoder, encode_errors, 'Encoding')
                    except BrokenPipeError as error:
                        raise RuntimeError('Encoding failed: ' + _read_error_log(encode_errors)) from error

                self._report_progress(0.9, 'Copying audio and subtitles...')
                # EOF fixes the exact video frame count. Limit copied tracks to
                # that duration, without extending video to match longer audio.
                duration = float(Fraction(processed, 1) / Fraction(rate))
                mux_command = [
                    'ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(encoded),
                    '-i', str(source), '-map', '0:v:0', '-map', '1:a?', '-map', '1:s?',
                    '-map_metadata', '1', '-c', 'copy', '-t', f'{duration:.9f}',
                    '-metadata:s:v:0', 'rotate=0', str(muxed),
                ]
                result = subprocess.run(mux_command, capture_output=True, **_process_options())
                if result.returncode:
                    raise RuntimeError('Muxing failed: ' + result.stderr.decode('utf-8', errors='replace')[-8000:])
                if not muxed.exists() or muxed.stat().st_size == 0:
                    raise RuntimeError('FFmpeg produced an empty output')
                os.replace(muxed, destination)
            self._report_progress(1.0, 'Video processing complete!')
            return True
        except Exception as error:
            self._report_progress(1.0, f'Error: {error}')
            print(f'Video processing error: {error}', file=sys.stderr)
            return False


def _process_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}


def _stop_process(process):
    """Reap child processes on success, exceptions, and KeyboardInterrupt."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _read_error_log(stream):
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 8000))
    return stream.read().decode('utf-8', errors='replace').strip()


def _check_process(process, errors, label):
    if process.wait(timeout=60):
        raise RuntimeError(f'{label} failed: {_read_error_log(errors)}')


def _read_frame(stream, byte_count):
    """Read a whole raw frame even if the pipe returns short reads."""
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == byte_count:
                return None
            raise ValueError('Truncated raw video frame')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _frame_batches(stream, frame_bytes, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            frame = _read_frame(stream, frame_bytes)
            if frame is None:
                break
            batch.append(frame)
        if not batch:
            return
        yield batch


def _process_frame_image(image, ditherer, method=None, max_size=64, multiplier=None, neural=None):
    if method == PixelizeMethod.NEURAL.value:
        image = (neural or NeuralPixelizer()).pixelize(image, max_size)
    elif method == PixelizeMethod.REGULAR.value:
        image = pixelize_regular(image, max_size)
    if ditherer is not None:
        image = ditherer.apply_dithering(image)
    # The encoder always requires even dimensions, including when resizing and
    # pixelization are both disabled.
    if multiplier or image.width % 2 or image.height % 2:
        image = _apply_final_resize_to_frame(image, multiplier or 1)
    return image.convert('RGB')


_FRAME_SETTINGS = None


def _init_frame_worker(size, output_size, ditherer, method, max_size, multiplier):
    global _FRAME_SETTINGS
    _FRAME_SETTINGS = (size, output_size, ditherer, method, max_size, multiplier)


def _process_stream_frame(data):
    size, output_size, ditherer, method, max_size, multiplier = _FRAME_SETTINGS
    image = Image.frombytes('RGB', size, data)
    result = _process_frame_image(image, ditherer, method, max_size, multiplier)
    if result.size != output_size:
        raise ValueError('Processed frame dimensions changed')
    return result.tobytes()


def _process_single_frame(frame_path: Path, ditherer: ImageDitherer,
                          pixelize_method: Optional[str] = None, max_size: int = 64,
                          final_resize_multiplier: Optional[int] = None) -> bool:
    """Compatibility worker for callers that already have a frame on disk."""
    try:
        with Image.open(frame_path) as source:
            image = source.convert('RGB')
        result = _process_frame_image(image, ditherer, pixelize_method, max_size, final_resize_multiplier)
        result.save(frame_path)
        return frame_path.exists() and frame_path.stat().st_size > 0
    except Exception as error:
        print(f'Error processing frame {frame_path}: {error}', file=sys.stderr)
        return False


def _apply_final_resize_to_frame(image: Image.Image, multiplier: int) -> Image.Image:
    """
    Apply final resize to image using integer multiplication and nearest-neighbor interpolation.
    This preserves the pixelated look when upscaling.
    Ensures dimensions are even for video codec compatibility.
    
    Args:
        image: The processed image to resize
        multiplier: Integer multiplier to scale the image by
        
    Returns:
        Resized image with even dimensions
    """
    current_w, current_h = image.size
    
    # Apply integer multiplication for perfect scaling
    new_w = current_w * multiplier
    new_h = current_h * multiplier
    
    # Ensure dimensions are even (required for libx264 yuv420p)
    if new_w % 2 != 0:
        new_w += 1
    if new_h % 2 != 0:
        new_h += 1
    
    # Use NEAREST to preserve pixelated look
    resized = image.resize((new_w, new_h), Image.Resampling.NEAREST)
    return resized


def _parse_video_info(metadata: dict) -> dict:
    streams = metadata.get('streams', [])
    if not streams:
        raise ValueError('Input has no video stream')
    stream = streams[0]
    try:
        width, height = int(stream['width']), int(stream['height'])
        if width <= 0 or height <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Video dimensions are missing or invalid') from error

    rate = None
    for field in ('avg_frame_rate', 'r_frame_rate'):
        try:
            candidate = Fraction(str(stream.get(field, '0/0')))
            if candidate > 0:
                rate = candidate
                break
        except (ValueError, ZeroDivisionError):
            continue
    if rate is None:
        raise ValueError('Video frame rate is missing or invalid')

    duration = None
    for value in (stream.get('duration'), metadata.get('format', {}).get('duration')):
        try:
            candidate = float(value)
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
                break
        except (ValueError, TypeError):
            continue
    try:
        frame_count = int(stream.get('nb_frames'))
        if frame_count <= 0:
            frame_count = None
    except (ValueError, TypeError):
        frame_count = None
    if frame_count is None and duration is not None:
        frame_count = max(1, round(duration * float(rate)))

    # FFmpeg autorotates decoded frames by default; report their display size.
    rotation = stream.get('tags', {}).get('rotate', 0)
    for data in stream.get('side_data_list', []):
        if 'rotation' in data:
            rotation = data['rotation']
            break
    try:
        if round(float(rotation)) % 180 in (90,):
            width, height = height, width
    except (TypeError, ValueError):
        pass
    return {'fps': float(rate), 'fps_fraction': str(rate), 'width': width,
            'height': height, 'duration': duration, 'frame_count': frame_count}
