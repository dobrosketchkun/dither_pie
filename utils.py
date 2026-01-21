"""
Utility functions for the dithering application.
"""

import json
import os
import requests
from typing import List, Tuple, Dict, Optional
from PIL import Image
import numpy as np

__all__ = [
    # Functions
    'load_palettes_from_file',
    'save_palettes_to_file',
    'hex_to_rgb',
    'rgb_to_hex',
    'palette_from_hex_list',
    'import_lospec_palette',
    'compute_even_dimensions',
    'estimate_video_memory_usage',
    'validate_video_file',
    'validate_image_file',
    'get_image_info',
    'ensure_rgb',
    # Classes
    'PaletteManager',
]


def load_palettes_from_file(filepath: str = "palette.json") -> List[Dict]:
    """
    Load custom palettes from JSON file.
    
    Args:
        filepath: Path to palette JSON file
        
    Returns:
        List of palette dictionaries with 'name' and 'colors' keys
    """
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            palettes = json.load(f)
        return palettes if isinstance(palettes, list) else []
    except Exception as e:
        print(f"Error loading palettes: {e}")
        return []


def save_palettes_to_file(palettes: List[Dict], filepath: str = "palette.json"):
    """
    Save palettes to JSON file.
    
    Args:
        palettes: List of palette dictionaries
        filepath: Path to save JSON file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(palettes, f, indent=4)
    except Exception as e:
        print(f"Error saving palettes: {e}")


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex string like "#FF0000" or "FF0000"
        
    Returns:
        RGB tuple (r, g, b)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.
    
    Args:
        rgb: RGB tuple (r, g, b)
        
    Returns:
        Hex string like "#FF0000"
    """
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


def palette_from_hex_list(hex_list: List[str]) -> List[Tuple[int, int, int]]:
    """
    Convert list of hex colors to palette (list of RGB tuples).
    
    Args:
        hex_list: List of hex color strings
        
    Returns:
        List of RGB tuples
    """
    return [hex_to_rgb(h) for h in hex_list]


def import_lospec_palette(url: str) -> Optional[Dict]:
    """
    Import a palette from lospec.com URL.
    
    Args:
        url: Lospec palette URL
        
    Returns:
        Dictionary with 'name' and 'colors' keys, or None if failed
    """
    try:
        # Extract palette slug from URL
        # e.g., https://lospec.com/palette-list/my-palette -> my-palette
        slug = url.rstrip('/').split('/')[-1]
        
        # Lospec API endpoint
        api_url = f"https://lospec.com/palette-list/{slug}.json"
        
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert hex colors to RGB tuples
        colors = [hex_to_rgb(f"#{c}") for c in data.get('colors', [])]
        
        if not colors:
            return None
        
        return {
            'name': data.get('name', slug),
            'colors': [rgb_to_hex(c) for c in colors]
        }
        
    except Exception as e:
        print(f"Error importing from Lospec: {e}")
        return None


def compute_even_dimensions(orig_w: int, orig_h: int, max_size: int) -> Tuple[int, int]:
    """
    Compute target dimensions such that the smaller side is close to max_size 
    and both dimensions are even (required for video codecs like libx264).
    
    Args:
        orig_w: Original width
        orig_h: Original height
        max_size: Target size for smaller dimension
        
    Returns:
        Tuple of (target_width, target_height)
    """
    if orig_w >= orig_h:
        # Landscape: use max_size as the target height
        target_h = max_size if max_size % 2 == 0 else max_size - 1
        target_w = int(round((orig_w / orig_h) * target_h))
        if target_w % 2 != 0:
            target_w += 1
    else:
        # Portrait: use max_size as the target width
        target_w = max_size if max_size % 2 == 0 else max_size - 1
        target_h = int(round((orig_h / orig_w) * target_w))
        if target_h % 2 != 0:
            target_h += 1
    
    return target_w, target_h


def estimate_video_memory_usage(width: int, height: int, frame_count: int) -> float:
    """
    Estimate memory usage for video processing in MB.
    
    Args:
        width: Video width
        height: Video height
        frame_count: Number of frames
        
    Returns:
        Estimated memory usage in megabytes
    """
    # Rough estimate: each frame is ~3 bytes per pixel (RGB) + overhead
    bytes_per_frame = width * height * 3 * 1.5  # 1.5x for overhead
    total_mb = (bytes_per_frame * frame_count) / (1024 * 1024)
    return total_mb


def validate_video_file(filepath: str) -> bool:
    """
    Check if file is a valid video file.
    
    Args:
        filepath: Path to video file
        
    Returns:
        True if valid video file
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    ext = os.path.splitext(filepath)[1].lower()
    return ext in video_extensions and os.path.exists(filepath)


def validate_image_file(filepath: str) -> bool:
    """
    Check if file is a valid image file.
    
    Args:
        filepath: Path to image file
        
    Returns:
        True if valid image file
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    ext = os.path.splitext(filepath)[1].lower()
    return ext in image_extensions and os.path.exists(filepath)


def get_image_info(filepath: str) -> Optional[Dict]:
    """
    Get basic image information.
    
    Args:
        filepath: Path to image file
        
    Returns:
        Dictionary with width, height, mode, format
    """
    try:
        with Image.open(filepath) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format
            }
    except Exception as e:
        print(f"Error getting image info: {e}")
        return None


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB mode.
    
    Args:
        image: PIL Image
        
    Returns:
        Image in RGB mode
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


class PaletteManager:
    """
    Manages custom palettes with loading, saving, and validation.
    """
    
    def __init__(self, filepath: str = "palette.json"):
        self.filepath = filepath
        self.palettes = []
        self.load()
    
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        return hex_to_rgb(hex_color)
    
    def load(self):
        """Load palettes from file."""
        self.palettes = load_palettes_from_file(self.filepath)
    
    def save(self):
        """Save palettes to file."""
        save_palettes_to_file(self.palettes, self.filepath)
    
    def add_palette(self, name: str, colors: List[str]):
        """Add a new palette."""
        # Check if palette with this name already exists
        for pal in self.palettes:
            if pal['name'] == name:
                # Update existing
                pal['colors'] = colors
                self.save()
                return
        
        # Add new
        self.palettes.append({'name': name, 'colors': colors})
        self.save()
    
    def remove_palette(self, name: str):
        """Remove a palette by name."""
        self.palettes = [p for p in self.palettes if p['name'] != name]
        self.save()
    
    def get_palette(self, name: str) -> Optional[Dict]:
        """Get palette by name."""
        for pal in self.palettes:
            if pal['name'] == name:
                return pal
        return None
    
    def get_palette_colors_rgb(self, name: str) -> Optional[List[Tuple[int, int, int]]]:
        """Get palette colors as RGB tuples."""
        pal = self.get_palette(name)
        if pal:
            return palette_from_hex_list(pal['colors'])
        return None
    
    def list_palette_names(self) -> List[str]:
        """Get list of all palette names."""
        return [p['name'] for p in self.palettes]

