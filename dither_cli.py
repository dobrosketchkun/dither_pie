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
from dithering_lib import DitherMode, ImageDitherer, ColorReducer
from video_processor import VideoProcessor, NeuralPixelizer, pixelize_regular
from utils import PaletteManager
from config_manager import ConfigManager
from PIL import Image


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
    
    # Rich handler for console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    handlers.append(rich_handler)
    
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
    
    def __enter__(self):
        """Setup progress bar."""
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
        if self.progress and self.task is not None:
            percentage = fraction * 100
            self.progress.update(self.task, completed=percentage, description=message)
    
    def finish(self):
        """Mark as complete."""
        if self.progress and self.task is not None:
            self.progress.update(self.task, completed=100, description="Complete!")


# ==================== Config Schema & Validation ====================

# Valid values for config fields
VALID_MODES = ["image", "video", "folder"]
VALID_PIXELIZATION_METHODS = ["none", "regular", "neural"]
VALID_PALETTE_SOURCES = ["median_cut", "kmeans", "uniform", "custom"]
VALID_DITHER_MODES = [mode.value for mode in DitherMode]


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """
    Validate configuration and return normalized config.
    
    Args:
        config: Raw config dictionary
        config_path: Path to config file (for resolving relative paths)
        
    Returns:
        Validated and normalized config
        
    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []
    
    # Required fields
    if "input" not in config:
        errors.append("Missing required field: 'input'")
    
    if "output" not in config:
        errors.append("Missing required field: 'output'")
    
    # Validate mode (optional, can be auto-detected)
    mode = config.get("mode")
    if mode and mode not in VALID_MODES:
        errors.append(f"Invalid mode: '{mode}'. Must be one of: {VALID_MODES}")
    
    # Validate pixelization section
    if "pixelization" in config:
        pix = config["pixelization"]
        if not isinstance(pix, dict):
            errors.append("'pixelization' must be an object/dictionary")
        else:
            if "method" in pix and pix["method"] not in VALID_PIXELIZATION_METHODS:
                errors.append(f"Invalid pixelization method: '{pix['method']}'. Must be one of: {VALID_PIXELIZATION_METHODS}")
            
            if "max_size" in pix:
                try:
                    max_size = int(pix["max_size"])
                    if max_size <= 0:
                        errors.append("'pixelization.max_size' must be positive")
                except (ValueError, TypeError):
                    errors.append("'pixelization.max_size' must be an integer")
    
    # Validate dithering section
    if "dithering" in config:
        dith = config["dithering"]
        if not isinstance(dith, dict):
            errors.append("'dithering' must be an object/dictionary")
        else:
            if "mode" in dith and dith["mode"] not in VALID_DITHER_MODES:
                errors.append(f"Invalid dither mode: '{dith['mode']}'. Must be one of: {VALID_DITHER_MODES}")
    
    # Validate palette section
    if "palette" in config:
        pal = config["palette"]
        if not isinstance(pal, dict):
            errors.append("'palette' must be an object/dictionary")
        else:
            if "source" in pal:
                source = pal["source"]
                # Can be a valid source, custom palette name, or "file:path"
                is_valid = (source in VALID_PALETTE_SOURCES or 
                           source.startswith("file:") or 
                           source.startswith("custom:"))
                
                # Also check if it's a palette name from palette.json
                if not is_valid:
                    try:
                        palette_mgr = PaletteManager()
                        palette_names = [p['name'] for p in palette_mgr.palettes]
                        if source in palette_names:
                            is_valid = True
                    except:
                        pass  # If palette manager fails, continue with validation
                
                if not is_valid:
                    errors.append(f"Invalid palette source: '{source}'")
            
            if "num_colors" in pal:
                try:
                    num_colors = int(pal["num_colors"])
                    if num_colors <= 0:
                        errors.append("'palette.num_colors' must be positive")
                except (ValueError, TypeError):
                    errors.append("'palette.num_colors' must be an integer")
    
    # Validate final_resize section
    if "final_resize" in config:
        resize = config["final_resize"]
        if not isinstance(resize, dict):
            errors.append("'final_resize' must be an object/dictionary")
        else:
            if "multiplier" in resize:
                try:
                    mult = int(resize["multiplier"])
                    if mult <= 0:
                        errors.append("'final_resize.multiplier' must be positive")
                except (ValueError, TypeError):
                    errors.append("'final_resize.multiplier' must be an integer")
    
    # If any errors, raise
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ConfigValidationError(error_msg)
    
    # Normalize paths (resolve relative to config file)
    config_dir = config_path.parent
    
    # Resolve input path
    input_path = Path(config["input"])
    if not input_path.is_absolute():
        input_path = (config_dir / input_path).resolve()
    config["input"] = str(input_path)
    
    # Resolve output path
    output_path = Path(config["output"])
    if not output_path.is_absolute():
        output_path = (config_dir / output_path).resolve()
    config["output"] = str(output_path)
    
    # Check if input exists
    if not Path(config["input"]).exists():
        raise ConfigValidationError(f"Input file/directory not found: {config['input']}")
    
    # Set defaults for optional fields
    config.setdefault("mode", None)  # Will be auto-detected
    config.setdefault("pixelization", {"enabled": False})
    config.setdefault("dithering", {"enabled": True, "mode": "bayer", "parameters": {}})
    config.setdefault("palette", {"source": "median_cut", "num_colors": 16, "use_gamma": False})
    config.setdefault("final_resize", {"enabled": False, "multiplier": 2})
    
    # Ensure nested defaults
    config["pixelization"].setdefault("enabled", False)
    config["pixelization"].setdefault("method", "regular")
    config["pixelization"].setdefault("max_size", 128)
    
    config["dithering"].setdefault("enabled", True)
    config["dithering"].setdefault("mode", "bayer")
    config["dithering"].setdefault("parameters", {})
    
    config["palette"].setdefault("source", "median_cut")
    config["palette"].setdefault("num_colors", 16)
    config["palette"].setdefault("use_gamma", False)
    
    config["final_resize"].setdefault("enabled", False)
    config["final_resize"].setdefault("multiplier", 2)
    
    return config


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Validated config dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in config file:\n  Line {e.lineno}: {e.msg}")
    except Exception as e:
        raise ConfigValidationError(f"Failed to load config file: {e}")
    
    # Validate and normalize
    return validate_config(config, config_path)


def detect_mode(input_path: Path) -> str:
    """
    Auto-detect processing mode based on input path.
    
    Args:
        input_path: Input file or directory path
        
    Returns:
        Mode string: "image", "video", or "folder"
    """
    if input_path.is_dir():
        return "folder"
    
    # Check file extension
    ext = input_path.suffix.lower()
    
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
    image_exts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    if ext in video_exts:
        return "video"
    elif ext in image_exts:
        return "image"
    else:
        raise ConfigValidationError(f"Cannot determine mode for file extension: {ext}")


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
    
    # Built-in palette generation methods (use num_colors from config)
    if source == "median_cut":
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.reduce_colors(source_image, num_colors)
        
    elif source == "kmeans":
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.generate_kmeans_palette(source_image, num_colors, random_state=42)
        
    elif source == "uniform":
        logger.info(f"Generating palette: [cyan]{source}[/] ({num_colors} colors)")
        palette = ColorReducer.generate_uniform_palette(num_colors)
        
    elif source.startswith("file:"):
        # Extract from another image file (use num_colors from config)
        file_path = source[5:]  # Remove "file:" prefix
        if not Path(file_path).exists():
            raise ConfigValidationError(f"Palette source image not found: {file_path}")
        
        logger.info(f"Extracting palette from: [cyan]{file_path}[/] ({num_colors} colors)")
        ref_image = Image.open(file_path).convert('RGB')
        palette = ColorReducer.generate_kmeans_palette(ref_image, num_colors, random_state=42)
        
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
        
        # Load image
        image = Image.open(input_path).convert('RGB')
        original_size = image.size
        logger.info(f"Image size: [cyan]{original_size[0]}x{original_size[1]}[/]")
        
        # Pixelization (optional)
        if config["pixelization"]["enabled"]:
            method = config["pixelization"]["method"]
            max_size = config["pixelization"]["max_size"]
            
            if method == "none":
                logger.info("Skipping pixelization")
                processed_image = image
            elif method == "regular":
                logger.info(f"Pixelizing (regular, max_size={max_size})...")
                processed_image = pixelize_regular(image, max_size)
                logger.info(f"[green]✓[/] Pixelized to {processed_image.size[0]}x{processed_image.size[1]}")
            elif method == "neural":
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
        logger.info(f"Saving to: [cyan]{output_path}[/]")
        processed_image.save(output_path)
        
        # Get file size
        file_size = output_path.stat().st_size
        size_kb = file_size / 1024
        logger.info(f"[bold green]✓ Image saved successfully![/] ({size_kb:.1f} KB)")
        
        return True
        
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
            
            if method == "regular":
                pixelize_func = ("regular", max_size)
            elif method == "neural":
                pixelize_func = ("neural", max_size)
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
            
    except Exception as e:
        logger.error(f"Failed to process video: {e}", exc_info=True)
        return False


def show_banner():
    """Display application banner."""
    banner = """
[bold cyan]╔═══════════════════════════════════════╗[/]
[bold cyan]║[/]      [bold white]Dither Pie CLI[/] [dim]- v1.0[/]        [bold cyan]║[/]
[bold cyan]║[/]  Image & Video Dithering Tool      [bold cyan]║[/]
[bold cyan]╚═══════════════════════════════════════╝[/]
"""
    console.print(banner)


def show_help():
    """Display detailed help information."""
    help_text = """
[bold cyan]Dither Pie CLI - Usage[/]

[bold]Basic Usage:[/]
  python dither_pie.py <config.json>        Process with JSON config
  python dither_pie.py --help               Show this help
  python dither_pie.py --example-config     Generate example config

[bold]Options:[/]
  --verbose, -v     Enable verbose output
  --quiet, -q       Suppress all but error messages
  --log-file FILE   Write log to file

[bold]Config File Format:[/]
  JSON file specifying input, output, and processing parameters.
  Use --example-config to generate a template.

[bold]Examples:[/]
  # Process single image
  python dither_pie.py configs/image_basic.json
  
  # Process video with verbose output
  python dither_pie.py -v configs/video.json
  
  # Batch process folder
  python dither_pie.py configs/batch_folder.json

[bold]Available Dither Modes:[/]
"""
    
    console.print(help_text)
    
    # List all dither modes
    console.print("  [bold]Dithering Algorithms:[/]")
    for mode in DitherMode:
        console.print(f"    • [cyan]{mode.value}[/]")
    
    console.print("\n[dim]For more info, see README_NEW.md[/]\n")


def generate_example_config():
    """Generate and print an example configuration file."""
    example = {
        "_comment": "Dither Pie CLI Configuration",
        "input": "path/to/input.png",
        "output": "path/to/output.png",
        "mode": "image",
        "pixelization": {
            "enabled": True,
            "method": "regular",
            "max_size": 128
        },
        "dithering": {
            "enabled": True,
            "mode": "bayer",
            "parameters": {}
        },
        "palette": {
            "_comment_source": "Options: median_cut, kmeans, uniform, file:path.png, custom:palette_name, or direct palette name",
            "source": "median_cut",
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
    example_json = json.dumps(example, indent=4)
    
    console.print("\n[bold cyan]Example Configuration:[/]\n")
    console.print(Panel(example_json, title="config.json", border_style="cyan"))
    console.print("\n[dim]Save this to a .json file and modify as needed.[/]\n")


def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Dither Pie CLI - Image & Video Dithering Tool",
        add_help=False  # We'll handle help ourselves
    )
    
    parser.add_argument('config', nargs='?', help='Path to JSON configuration file')
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
        show_banner()
        generate_example_config()
        sys.exit(0)
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    
    # Show banner (unless quiet mode)
    if not args.quiet:
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
    
    # Load and validate config
    try:
        config = load_config(config_path)
    except ConfigValidationError as e:
        logger.error(f"[bold red]{e}[/]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        sys.exit(1)
    
    logger.info("[green]✓[/] Configuration validated")
    
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
    
    if mode == "image":
        success = process_single_image(config)
        
    elif mode == "video":
        success = process_single_video(config)
        
    elif mode == "folder":
        # Folder processing will be implemented in Phase 5
        logger.error("Folder batch processing not yet implemented (Phase 5)")
        success = False
    
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

