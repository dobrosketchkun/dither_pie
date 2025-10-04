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
            "source": "median_cut",
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
    
    # TODO: Load and validate config
    # TODO: Process based on mode (image/video/folder)
    
    logger.info("[bold green]✓[/] Processing complete!")


if __name__ == "__main__":
    main()

