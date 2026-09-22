#!/usr/bin/env python3
"""
CLI module for Dither Pie - Command-Line Interface

Provides command-line interface for batch processing images and videos
with dithering effects. Uses Rich for beautiful terminal output.
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

# Local imports
from dithering_lib import DitherMode, ImageDitherer, ColorReducer, PixelizeMethod, PaletteSource
from video_processor import VideoProcessor
from pixelization import NeuralPixelizer, pixelize_regular
from processing_config import (
    ConfigValidationError, validate_config, load_config, detect_mode,
    VALID_MODES, VALID_PIXELIZATION_METHODS, VALID_PALETTE_SOURCES, VALID_DITHER_MODES,
)
from utils import (
    PaletteManager,
    has_actual_transparency,
    check_save_format,
    split_alpha,
)
from config_manager import ConfigManager
from PIL import Image

__all__ = [
    'main',
    'setup_logging',
    'CLIProgressCallback',
    'ConfigValidationError',
    'validate_config',
    'load_config',
    'detect_mode',
    'process_single_image',
    'process_single_video',
    'process_folder',
]


# Initialize Rich console
console = Console()

# Logger instance
logger = None


def setup_logging(verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None):
    """
    Setup logging with Rich handler for beautiful terminal output.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        quiet: Suppress all but ERROR messages
        log_file: Optional path to log file
    """
    global logger
    
    # Determine logging level
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure logging
    handlers = []
    
    # Use Rich handler only if stdout is a real terminal
    # Otherwise use plain StreamHandler to avoid Unicode encoding issues
    if sys.stdout.isatty():
        # Rich handler for console output
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        handlers.append(rich_handler)
    else:
        # Plain handler for redirected output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        handlers.append(stream_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    # Setup root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers
    )
    
    logger = logging.getLogger('dither_pie')
    logger.setLevel(level)
    
    return logger


class CLIProgressCallback:
    """
    Progress callback for video processing that uses Rich progress bars.
    Compatible with VideoProcessor.progress_callback signature.
    """
    
    def __init__(self, total_frames: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            total_frames: Total number of frames (for progress calculation)
        """
        self.total_frames = total_frames
        self.progress = None
        self.task = None
        self.use_rich = sys.stdout.isatty()  # Only use Rich if terminal
    
    def __enter__(self):
        """Setup progress bar."""
        if self.use_rich:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            )
            self.progress.__enter__()
            self.task = self.progress.add_task("Processing video...", total=100)
        return self
    
    def __exit__(self, *args):
        """Cleanup progress bar."""
        if self.progress:
            self.progress.__exit__(*args)
    
    def update(self, fraction: float, message: str):
        """
        Update progress bar.
        
        Args:
            fraction: Progress fraction (0.0 to 1.0)
            message: Status message
        """
        if self.use_rich and self.progress and self.task is not None:
            percentage = fraction * 100
            self.progress.update(self.task, completed=percentage, description=message)
        elif not self.use_rich:
            # Plain text progress for redirected output
            percentage = int(fraction * 100)
            print(f"Progress: {percentage}% - {message}", flush=True)
    
    def finish(self):
        """Mark as complete."""
        if self.use_rich and self.progress and self.task is not None:
            self.progress.update(self.task, completed=100, description="Complete!")
        elif not self.use_rich:
            print("Progress: 100% - Complete!", flush=True)


# ==================== Palette Setup ====================

def setup_palette_from_config(palette_config: Dict[str, Any], source_image: Image.Image) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Setup palette based on configuration.
    
    Args:
        palette_config: Palette configuration from config
        source_image: Source image for palette generation
        
    Returns:
        Tuple of (palette as list of RGB tuples, actual color count to use)
    """
    source = palette_config["source"]
    num_colors = palette_config["num_colors"]

    # Check if this is a custom/predefined palette (ignore num_colors)
    is_custom_palette = False

    # The palette is built here and handed to ImageDitherer, which then skips
    # its own generation - so the opaque mask has to be applied at this level
    # too, or invisible pixels would still shape the palette.
    source_rgb, source_alpha = split_alpha(source_image)
    source_mask = None if source_alpha is None else (source_alpha > 0)

    # Built-in palette generation methods (use num_colors from config)
    if source == PaletteSource.MEDIAN_CUT.value:
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.reduce_colors(source_rgb, num_colors, mask=source_mask)

    elif source == PaletteSource.KMEANS.value:
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.generate_kmeans_palette(
            source_rgb, num_colors, random_state=42, mask=source_mask
        )

    elif source == PaletteSource.UNIFORM.value:
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.generate_uniform_palette(num_colors)
        
    elif source.startswith("file:"):
        # Extract from another image file (use num_colors from config)
        file_path = source[5:]  # Remove "file:" prefix
        if not Path(file_path).exists():
            raise ConfigValidationError(f"Palette source image not found: {file_path}")
        
        logger.info(f"Extracting palette from: [cyan]{file_path}[/] ({num_colors} colors)")
        # A reference image's invisible pixels should not shape the palette any
        # more than the target image's should.
        ref_image, ref_alpha = split_alpha(Image.open(file_path))
        ref_mask = None if ref_alpha is None else (ref_alpha > 0)
        if ref_mask is not None:
            logger.info("[dim]Reference has transparency - extracting from opaque pixels only[/]")
        palette = ColorReducer.generate_kmeans_palette(
            ref_image, num_colors, random_state=42, mask=ref_mask
        )
        
    elif source.startswith("custom:"):
        # Load from palette.json (use palette's actual color count)
        palette_name = source[7:]  # Remove "custom:" prefix
        palette_mgr = PaletteManager()
        
        # Find the palette
        found = False
        for palette_data in palette_mgr.palettes:
            if palette_data['name'] == palette_name:
                hex_colors = palette_data['colors']
                palette = [palette_mgr._hex_to_rgb(c) for c in hex_colors]
                is_custom_palette = True
                found = True
                break
        
        if not found:
            raise ConfigValidationError(f"Custom palette not found: {palette_name}")
        
        logger.info(f"Loading custom palette: [cyan]{palette_name}[/] ({len(palette)} colors)")
            
    else:
        # Try to load as custom palette name directly from palette.json
        palette_mgr = PaletteManager()
        found = False
        for palette_data in palette_mgr.palettes:
            if palette_data['name'] == source:
                hex_colors = palette_data['colors']
                palette = [palette_mgr._hex_to_rgb(c) for c in hex_colors]
                is_custom_palette = True
                found = True
                break
        
        if not found:
            raise ConfigValidationError(f"Unknown palette source: {source}")
        
        logger.info(f"Loading custom palette: [cyan]{source}[/] ({len(palette)} colors)")
    
    # Use actual palette color count for custom palettes, config num_colors otherwise
    actual_num_colors = len(palette) if is_custom_palette else num_colors
    
    logger.info(f"[green]✓[/] Palette ready with {len(palette)} colors")
    return palette, actual_num_colors


# ==================== Image Processing ====================

def process_single_image(config: Dict[str, Any]) -> bool:
    """
    Process a single image with pixelization and dithering.
    
    Args:
        config: Validated configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        input_path = Path(config["input"])
        output_path = Path(config["output"])
        
        logger.info(f"Loading image: [cyan]{input_path.name}[/]")
        
        # Load image. Only images with genuinely non-opaque pixels are kept as
        # RGBA; a fully-opaque RGBA PNG becomes RGB exactly as before.
        loaded = Image.open(input_path)
        image_has_alpha = has_actual_transparency(loaded)
        image = loaded.convert('RGBA') if image_has_alpha else loaded.convert('RGB')
        original_size = image.size
        logger.info(f"Image size: [cyan]{original_size[0]}x{original_size[1]}[/]")
        if image_has_alpha:
            logger.info("[cyan]Transparency detected[/] - alpha will be preserved")

        # Pixelization (optional)
        if config["pixelization"]["enabled"]:
            method = config["pixelization"]["method"]
            max_size = config["pixelization"]["max_size"]

            if method == PixelizeMethod.NONE.value:
                logger.info("Skipping pixelization")
                processed_image = image
            elif method == PixelizeMethod.REGULAR.value:
                logger.info(f"Pixelizing (regular, max_size={max_size})...")
                processed_image = pixelize_regular(image, max_size)
                logger.info(f"[green]✓[/] Pixelized to {processed_image.size[0]}x{processed_image.size[1]}")
            elif method == PixelizeMethod.NEURAL.value:
                # The neural model is 3-channel and cannot carry alpha. Refuse
                # rather than silently flattening, matching the GUI which
                # disables the button outright.
                if image_has_alpha:
                    logger.error(
                        f"Neural pixelization cannot preserve transparency, and "
                        f"'{input_path.name}' has non-opaque pixels. "
                        f"Use \"method\": \"regular\" instead."
                    )
                    return False
                logger.info(f"Pixelizing (neural, max_size={max_size})... [dim](this may take a moment)[/]")
                neural_pix = NeuralPixelizer()
                processed_image = neural_pix.pixelize(image, max_size)
                logger.info(f"[green]✓[/] Neural pixelization complete: {processed_image.size[0]}x{processed_image.size[1]}")
            else:
                processed_image = image
        else:
            processed_image = image
        
        # Dithering
        if config["dithering"]["enabled"]:
            dither_mode_str = config["dithering"]["mode"]
            dither_params = config["dithering"].get("parameters", {})
            
            logger.info(f"Applying dithering: [cyan]{dither_mode_str}[/]")
            
            # Setup palette (returns palette and actual color count)
            palette, actual_num_colors = setup_palette_from_config(config["palette"], processed_image)
            
            # Parse dither mode
            try:
                dither_mode = DitherMode(dither_mode_str)
            except ValueError:
                logger.error(f"Invalid dither mode: {dither_mode_str}")
                return False
            
            # Create ditherer (use actual_num_colors from palette)
            ditherer = ImageDitherer(
                num_colors=actual_num_colors,
                dither_mode=dither_mode,
                palette=palette,
                use_gamma=config["palette"]["use_gamma"],
                dither_params=dither_params
            )
            
            # Apply dithering
            processed_image = ditherer.apply_dithering(processed_image)
            logger.info(f"[green]✓[/] Dithering complete")
        
        # Final resize (optional)
        if config["final_resize"]["enabled"]:
            multiplier = config["final_resize"]["multiplier"]
            logger.info(f"Applying final resize (×{multiplier})...")
            
            w, h = processed_image.size
            new_w, new_h = w * multiplier, h * multiplier
            processed_image = processed_image.resize((new_w, new_h), Image.Resampling.NEAREST)
            logger.info(f"[green]✓[/] Resized to {new_w}x{new_h}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save result
        problem = check_save_format(processed_image, output_path)
        if problem:
            logger.error(problem)
            return False
        logger.info(f"Saving to: [cyan]{output_path}[/]")
        processed_image.save(output_path)
        
        # Get file size
        file_size = output_path.stat().st_size
        size_kb = file_size / 1024
        logger.info(f"[bold green]✓ Image saved successfully![/] ({size_kb:.1f} KB)")
        
        return True
    
    except KeyboardInterrupt:
        logger.warning("\n[yellow]Image processing interrupted by user[/]")
        raise  # Re-raise to stop execution
        
    except Exception as e:
        logger.error(f"Failed to process image: {e}", exc_info=True)
        return False


# ==================== Video Processing ====================

def process_single_video(config: Dict[str, Any], neural_pixelizer: Optional[NeuralPixelizer] = None) -> bool:
    """
    Process a single video with pixelization and dithering.
    
    Args:
        config: Validated configuration dictionary
        neural_pixelizer: Optional pre-loaded neural pixelizer (for batch processing)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        input_path = Path(config["input"])
        output_path = Path(config["output"])
        
        logger.info(f"Processing video: [cyan]{input_path.name}[/]")
        
        # Get video info
        video_processor = VideoProcessor()
        try:
            video_info = video_processor.get_video_info(str(input_path))
            logger.info(f"Video: {video_info['width']}x{video_info['height']}, "
                       f"{video_info['fps']:.2f} fps, {video_info['frame_count']} frames")
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
        
        # Load first frame for palette generation
        logger.info("Loading first frame for palette generation...")
        import tempfile
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = os.path.join(tmpdir, "frame.png")
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-vframes", "1",
                frame_path,
                "-y"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            first_frame = Image.open(frame_path).convert('RGB')
        
        # Setup palette (returns palette and actual color count)
        palette, actual_num_colors = setup_palette_from_config(config["palette"], first_frame)
        
        # Setup ditherer
        dither_mode_str = config["dithering"]["mode"]
        dither_params = config["dithering"].get("parameters", {})
        
        try:
            dither_mode = DitherMode(dither_mode_str)
        except ValueError:
            logger.error(f"Invalid dither mode: {dither_mode_str}")
            return False
        
        ditherer = ImageDitherer(
            num_colors=actual_num_colors,
            dither_mode=dither_mode,
            palette=palette,
            use_gamma=config["palette"]["use_gamma"],
            dither_params=dither_params
        )
        
        # Setup pixelization function
        pixelize_func = None
        if config["pixelization"]["enabled"]:
            method = config["pixelization"]["method"]
            max_size = config["pixelization"]["max_size"]
            
            if method == PixelizeMethod.REGULAR.value:
                pixelize_func = (PixelizeMethod.REGULAR.value, max_size)
            elif method == PixelizeMethod.NEURAL.value:
                pixelize_func = (PixelizeMethod.NEURAL.value, max_size)
                # Pre-load neural pixelizer if not provided
                if neural_pixelizer is None:
                    logger.info("Loading neural pixelization models...")
                    neural_pixelizer = NeuralPixelizer()
        
        # Setup final resize
        final_resize_multiplier = None
        if config["final_resize"]["enabled"]:
            final_resize_multiplier = config["final_resize"]["multiplier"]
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Progress callback
        progress_callback = CLIProgressCallback()
        video_processor = VideoProcessor(progress_callback=lambda f, m: progress_callback.update(f, m))
        
        # Process video
        logger.info("Processing video frames...")
        with progress_callback:
            success = video_processor.process_video_streaming(
                str(input_path),
                str(output_path),
                ditherer,
                pixelize_func=pixelize_func,
                final_resize_multiplier=final_resize_multiplier
            )
        
        if success:
            # Get file size
            file_size = output_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            logger.info(f"[bold green]✓ Video processed successfully![/] ({size_mb:.1f} MB)")
            return True
        else:
            logger.error("Video processing failed")
            return False
    
    except KeyboardInterrupt:
        logger.warning("\n[yellow]Video processing interrupted by user[/]")
        raise  # Re-raise to stop execution
            
    except Exception as e:
        logger.error(f"Failed to process video: {e}", exc_info=True)
        return False


# ==================== Filename Generation ====================

def generate_output_filename(input_path: Path, config: Dict[str, Any]) -> Path:
    """
    Generate smart output filename based on input and config settings.
    
    Args:
        input_path: Input file path
        config: Configuration dictionary
        
    Returns:
        Output path in the same directory as input with descriptive name
    """
    # Get base filename (cap at 30 chars to prevent long names)
    base_stem = input_path.stem
    if len(base_stem) > 30:
        base_stem = base_stem[:30]
    
    parts = [base_stem]
    
    # Add pixelization info if enabled
    if config["pixelization"]["enabled"]:
        method = config["pixelization"]["method"]
        if method != PixelizeMethod.NONE.value:
            parts.append(f"pix{config['pixelization']['max_size']}")
    
    # Add dithering info if enabled
    if config["dithering"]["enabled"]:
        parts.append(config["dithering"]["mode"])
        
        # Add palette info
        palette_source = config["palette"]["source"]
        num_colors = config["palette"]["num_colors"]
        
        # Simplify palette source for filename
        if palette_source == PaletteSource.MEDIAN_CUT.value:
            parts.append(f"{num_colors}c")
        elif palette_source == PaletteSource.KMEANS.value:
            parts.append(f"km{num_colors}c")
        elif palette_source == PaletteSource.UNIFORM.value:
            parts.append(f"uni{num_colors}c")
        elif palette_source.startswith(f"{PaletteSource.FROM_FILE.value}:"):
            parts.append(f"{num_colors}c")
        else:
            # Custom palette name (cap at 10 chars)
            palette_name = palette_source.replace("custom:", "")
            palette_name = palette_name[:10] if len(palette_name) > 10 else palette_name
            parts.append(palette_name)
        
        # Add gamma if enabled
        if config["palette"]["use_gamma"]:
            parts.append("gamma")
    
    # Create output filename
    output_stem = "_".join(parts)
    output_path = input_path.parent / f"{output_stem}{input_path.suffix}"
    
    return output_path


# ==================== Batch Folder Processing ====================

def process_folder(config: Dict[str, Any]) -> bool:
    """
    Process all images or videos in a folder.
    
    Args:
        config: Validated configuration dictionary
        
    Returns:
        True only if every discovered file processed successfully, False otherwise
    """
    try:
        input_path = Path(config["input"])
        output_path = Path(config["output"])
        
        if not input_path.is_dir():
            logger.error(f"Input path is not a directory: {input_path}")
            return False
        
        # Determine file type based on first few files or config hint
        image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        
        # Scan directory for processable files
        all_files = list(input_path.iterdir())
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in image_exts]
        video_files = [f for f in all_files if f.is_file() and f.suffix.lower() in video_exts]
        
        # Determine what to process
        process_images = len(image_files) > 0
        process_videos = len(video_files) > 0
        
        if not process_images and not process_videos:
            logger.error(f"No processable image or video files found in: {input_path}")
            return False
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_files = len(image_files) + len(video_files)
        logger.info(f"Found {len(image_files)} images and {len(video_files)} videos")
        logger.info(f"Output directory: [cyan]{output_path}[/]")
        logger.info("")
        
        # Track results
        success_count = 0
        failed_count = 0
        failed_files = []
        
        # Pre-load neural pixelizer if needed (for performance)
        neural_pixelizer = None
        if config["pixelization"]["enabled"] and config["pixelization"]["method"] == PixelizeMethod.NEURAL.value:
            logger.info("Pre-loading neural pixelization models... [dim](one-time setup)[/]")
            try:
                neural_pixelizer = NeuralPixelizer()
                logger.info("[green]✓[/] Neural models loaded")
            except Exception as e:
                logger.error(f"Failed to load neural models: {e}")
                return False
        
        # Process images
        if process_images:
            logger.info(f"[bold cyan]Processing {len(image_files)} images...[/]")
            logger.info("")
            
            for idx, image_file in enumerate(image_files, 1):
                try:
                    # Generate output filename
                    output_file = output_path / image_file.name
                    
                    # Create config for this file
                    file_config = config.copy()
                    file_config["input"] = str(image_file)
                    file_config["output"] = str(output_file)
                    file_config["mode"] = "image"
                    
                    logger.info(f"[{idx}/{len(image_files)}] Processing: [cyan]{image_file.name}[/]")
                    
                    # Process the image
                    success = process_single_image(file_config)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_files.append(image_file.name)
                    
                    logger.info("")  # Empty line between files
                    
                except KeyboardInterrupt:
                    logger.warning("\n[yellow]Processing interrupted by user[/]")
                    raise
                except Exception as e:
                    logger.error(f"Error processing {image_file.name}: {e}")
                    failed_count += 1
                    failed_files.append(image_file.name)
                    logger.info("")
        
        # Process videos
        if process_videos:
            logger.info(f"[bold cyan]Processing {len(video_files)} videos...[/]")
            logger.info("")
            
            for idx, video_file in enumerate(video_files, 1):
                try:
                    # Generate output filename
                    output_file = output_path / video_file.name
                    
                    # Create config for this file
                    file_config = config.copy()
                    file_config["input"] = str(video_file)
                    file_config["output"] = str(output_file)
                    file_config["mode"] = "video"
                    
                    logger.info(f"[{idx}/{len(video_files)}] Processing: [cyan]{video_file.name}[/]")
                    
                    # Process the video (pass pre-loaded neural pixelizer)
                    success = process_single_video(file_config, neural_pixelizer=neural_pixelizer)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_files.append(video_file.name)
                    
                    logger.info("")  # Empty line between files
                    
                except KeyboardInterrupt:
                    logger.warning("\n[yellow]Processing interrupted by user[/]")
                    raise
                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")
                    failed_count += 1
                    failed_files.append(video_file.name)
                    logger.info("")
        
        # Summary
        logger.info("[bold]═══════════════════════════════════════[/]")
        logger.info(f"[bold]Batch Processing Summary[/]")
        logger.info("[bold]═══════════════════════════════════════[/]")
        logger.info(f"Total files:     {total_files}")
        logger.info(f"[green]Successful:[/]     {success_count}")
        if failed_count > 0:
            logger.info(f"[red]Failed:[/]         {failed_count}")
            logger.info("")
            logger.info("[red]Failed files:[/]")
            for failed_file in failed_files:
                logger.info(f"  • {failed_file}")
        
        return failed_count == 0 and success_count == total_files
        
    except Exception as e:
        logger.error(f"Failed to process folder: {e}", exc_info=True)
        return False



def show_banner():
    """Display application banner."""
    banner = """
[bold cyan]░░░░       ░░░        ░░        ░░  ░░░░  ░░        ░░       ░░░░░░░░░       ░░░        ░░        ░░░░░░░░░      ░░░  ░░░░░░░░        ░░░░[/]
[bold cyan]▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒[/]
[bold cyan]▓▓▓▓  ▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓▓        ▓▓      ▓▓▓▓       ▓▓▓▓▓▓▓▓▓       ▓▓▓▓▓▓  ▓▓▓▓▓      ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓[/]
[bold cyan]████  ████  █████  ████████  █████  ████  ██  ████████  ███  █████████  ███████████  █████  ██████████████  ████  ██  ███████████  ███████[/]
[bold cyan]████       ███        █████  █████  ████  ██        ██  ████  ████████  ████████        ██        █████████      ███        ██        ████[/]
"""
    console.print(banner)



def show_help():
    """Display detailed help information."""
    help_text = """
[bold cyan]Dither Pie CLI - Usage[/]

[bold]Basic Usage:[/]
  python dither_pie.py <config.json>                    Process with JSON config
  python dither_pie.py <config.json> <file/folder>      Process file/folder with config settings
  python dither_pie.py --help                           Show this help
  python dither_pie.py --example-config                 Generate example config

[bold]Options:[/]
  --verbose, -v     Enable verbose output
  --quiet, -q       Suppress all but error messages
  --log-file FILE   Write log to file

[bold]Config File Format:[/]
  JSON file specifying processing parameters (input/output optional when using second argument).
  Use --example-config to generate a template.

[bold]Examples:[/]
  # Process single image with config
  python dither_pie.py configs/image_basic.json
  
  # Process specific file (generates smart output name)
  python dither_pie.py configs/settings.json photo.jpg
  # Output: photo_bayer_16c.jpg (in same directory)
  
  # Process folder (creates output_processed folder)
  python dither_pie.py configs/settings.json images/
  # Output: images_processed/ folder
  
  # Process video with verbose output
  python dither_pie.py -v configs/video.json
  
  # Batch process with override
  python dither_pie.py configs/batch.json my_photos/

[bold]Available Dither Modes:[/]
"""
    
    console.print(help_text)
    
    # List all dither modes
    console.print("  [bold]Dithering Algorithms:[/]")
    for mode in DitherMode:
        console.print(f"    • [cyan]{mode.value}[/]")
    
    console.print("\n[dim]For more info, see README.md[/]\n")


def generate_example_config():
    """Generate and print an example configuration file."""
    example = {
        "_comment": "Dither Pie CLI Configuration",
        "input": "path/to/input.png",
        "output": "path/to/output.png",
        "mode": "image",
        "pixelization": {
            "enabled": True,
            "method": PixelizeMethod.REGULAR.value,
            "max_size": 128
        },
        "dithering": {
            "enabled": True,
            "mode": "bayer",
            "parameters": {}
        },
        "palette": {
            "_comment_source": "Options: median_cut, kmeans, uniform, file:path.png, custom:palette_name, or direct palette name",
            "source": PaletteSource.MEDIAN_CUT.value,
            "_comment_num_colors": "Ignored for custom/predefined palettes (uses palette's actual color count)",
            "num_colors": 16,
            "use_gamma": False
        },
        "final_resize": {
            "enabled": False,
            "multiplier": 2
        }
    }
    
    import json
    # Print clean JSON for piping/redirection
    print(json.dumps(example, indent=4))


def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Dither Pie CLI - Image & Video Dithering Tool",
        add_help=False  # We'll handle help ourselves
    )
    
    parser.add_argument('config', nargs='?', help='Path to JSON configuration file')
    parser.add_argument('input_override', nargs='?', help='Optional: file/folder to process (overrides config input/output)')
    parser.add_argument('--help', '-h', action='store_true', help='Show help')
    parser.add_argument('--example-config', action='store_true', help='Generate example config')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (errors only)')
    parser.add_argument('--log-file', type=str, help='Log to file')
    
    args = parser.parse_args()
    
    # Handle special commands first (before logging setup)
    if args.help:
        show_banner()
        show_help()
        sys.exit(0)
    
    if args.example_config:
        # Don't show banner for example config - output should be clean JSON
        generate_example_config()
        sys.exit(0)
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    
    # Show banner (unless quiet mode or output is redirected)
    if not args.quiet and sys.stdout.isatty():
        show_banner()
    
    # Check if config file provided
    if not args.config:
        console.print("[bold red]Error:[/] No configuration file specified.\n")
        console.print("Usage: python dither_pie.py <config.json>")
        console.print("       python dither_pie.py --help\n")
        sys.exit(1)
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading configuration from: [cyan]{config_path}[/]")
    
    # Load and validate config (skip input check if we have an override)
    try:
        config = load_config(config_path, skip_input_check=bool(args.input_override))
    except ConfigValidationError as e:
        logger.error(f"[bold red]{e}[/]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        sys.exit(1)
    
    logger.info("[green]✓[/] Configuration validated")
    
    # Handle input override (second argument)
    if args.input_override:
        input_override_path = Path(args.input_override)
        
        # Check if it exists
        if not input_override_path.exists():
            logger.error(f"Input override file/folder not found: {input_override_path}")
            sys.exit(1)
        
        # Override input
        config["input"] = str(input_override_path.resolve())
        
        # Generate smart output filename
        if input_override_path.is_dir():
            # For folders, create output folder in same parent directory
            output_folder_name = f"{input_override_path.name}_processed"
            config["output"] = str((input_override_path.parent / output_folder_name).resolve())
            config["mode"] = "folder"
        else:
            # For files, generate smart filename in same directory
            output_file = generate_output_filename(input_override_path, config)
            config["output"] = str(output_file.resolve())
            config["mode"] = None  # Will be auto-detected
        
        logger.info(f"[cyan]Using input override:[/] {input_override_path.name}")
        logger.info(f"[cyan]Generated output:[/] {Path(config['output']).name}")
    
    # Auto-detect mode if not specified
    if not config["mode"]:
        input_path = Path(config["input"])
        try:
            config["mode"] = detect_mode(input_path)
            logger.info(f"Auto-detected mode: [cyan]{config['mode']}[/]")
        except ConfigValidationError as e:
            logger.error(f"{e}")
            sys.exit(1)
    
    # Log processing summary
    logger.info(f"Input:  [cyan]{config['input']}[/]")
    logger.info(f"Output: [cyan]{config['output']}[/]")
    logger.info(f"Mode:   [cyan]{config['mode']}[/]")
    
    if config["pixelization"]["enabled"]:
        logger.info(f"Pixelization: [yellow]{config['pixelization']['method']}[/] (max_size={config['pixelization']['max_size']})")
    else:
        logger.info("Pixelization: [dim]disabled[/]")
    
    if config["dithering"]["enabled"]:
        logger.info(f"Dithering: [yellow]{config['dithering']['mode']}[/]")
    else:
        logger.info("Dithering: [dim]disabled[/]")
    
    logger.info(f"Palette: [yellow]{config['palette']['source']}[/] ({config['palette']['num_colors']} colors)")
    
    # Process based on mode
    logger.info("")  # Empty line for readability
    
    mode = config["mode"]
    success = False
    
    try:
        if mode == "image":
            success = process_single_image(config)
            
        elif mode == "video":
            success = process_single_video(config)
            
        elif mode == "folder":
            success = process_folder(config)
    
    except KeyboardInterrupt:
        logger.info("")
        logger.warning("[bold yellow]⚠ Processing interrupted by user (Ctrl+C)[/]")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    # Exit with appropriate code
    if success:
        logger.info("")
        logger.info("[bold green]✓ Processing complete![/]")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("[bold red]✗ Processing failed![/]")
        sys.exit(1)


if __name__ == "__main__":
    main()

