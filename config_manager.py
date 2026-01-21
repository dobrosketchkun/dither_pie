"""
Configuration management for the dithering application.
Handles loading, saving, and managing user preferences.
"""

import json
import os
from typing import Any, Optional, Dict
from pathlib import Path

__all__ = [
    'ConfigManager',
]


class ConfigManager:
    """Manages application configuration and user preferences."""
    
    DEFAULT_CONFIG = {
        # Window settings
        "window": {
            "width": 1400,
            "height": 900,
            "x": None,  # None means center on screen
            "y": None,
            "maximized": False
        },
        
        # Theme settings
        "theme": {
            "appearance_mode": "system",  # "system", "dark", "light"
            "color_theme": "blue"
        },
        
        # Default processing settings
        "defaults": {
            "max_size": 640,
            "num_colors": 16,
            "dither_mode": "bayer",
            "use_gamma": False,
            "final_resize_enabled": False,
            "final_size": 1920
        },
        
        # Last used paths
        "paths": {
            "last_image_dir": None,
            "last_video_dir": None,
            "last_save_dir": None
        },
        
        # UI component sizes
        "ui": {
            "sidebar_width": 280,
            "palette_dialog_width": 400,
            "palette_dialog_height": 600,
            "palette_dialog_x": None,  # None means center on parent
            "palette_dialog_y": None,
            "spinner_name": "dots"  # Spinner animation style (from spinners.json)
        },
        
        # Recent files (keep last 10)
        "recent_files": []
    }
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize config manager.
        
        Args:
            config_file: Path to config file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load config from file, or create default if not exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Merge with defaults to handle new settings
                return self._merge_configs(self.DEFAULT_CONFIG.copy(), loaded)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config
            self.save()
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        Recursively merge loaded config with defaults.
        Ensures all default keys exist even if not in loaded config.
        """
        for key, value in default.items():
            if key in loaded:
                if isinstance(value, dict) and isinstance(loaded[key], dict):
                    default[key] = self._merge_configs(value, loaded[key])
                else:
                    default[key] = loaded[key]
        return default
    
    def save(self):
        """Save current config to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get config value by nested keys.
        
        Args:
            *keys: Nested keys (e.g., "window", "width")
            default: Default value if key not found
            
        Returns:
            Config value or default
            
        Example:
            config.get("window", "width")  # Returns 1400
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys: str, value: Any):
        """
        Set config value by nested keys.
        
        Args:
            *keys: Nested keys (e.g., "window", "width")
            value: Value to set
            
        Example:
            config.set("window", "width", value=1600)
        """
        if len(keys) == 0:
            return
        
        # Navigate to the parent dict
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_window_geometry(self) -> str:
        """
        Get window geometry string for tkinter.
        
        Returns:
            Geometry string like "1400x900" or "1400x900+100+50"
        """
        w = self.get("window", "width", default=1400)
        h = self.get("window", "height", default=900)
        x = self.get("window", "x")
        y = self.get("window", "y")
        
        if x is not None and y is not None:
            return f"{w}x{h}+{x}+{y}"
        else:
            return f"{w}x{h}"
    
    def save_window_geometry(self, geometry_string: str, maximized: bool = False):
        """
        Save window geometry from tkinter geometry string.
        
        Args:
            geometry_string: Tkinter geometry string like "1400x900+100+50"
            maximized: Whether window is maximized
        """
        try:
            # Parse geometry string
            size_pos = geometry_string.split('+')
            size = size_pos[0].split('x')
            
            self.set("window", "width", value=int(size[0]))
            self.set("window", "height", value=int(size[1]))
            
            if len(size_pos) >= 3:
                self.set("window", "x", value=int(size_pos[1]))
                self.set("window", "y", value=int(size_pos[2]))
            
            self.set("window", "maximized", value=maximized)
        except Exception as e:
            print(f"Error parsing geometry: {e}")
    
    def update_last_path(self, path_type: str, filepath: str):
        """
        Update last used directory for a path type.
        
        Args:
            path_type: "image", "video", or "save"
            filepath: File path to extract directory from
        """
        if filepath:
            directory = str(Path(filepath).parent)
            self.set("paths", f"last_{path_type}_dir", value=directory)
    
    def get_last_path(self, path_type: str) -> Optional[str]:
        """
        Get last used directory for a path type.
        
        Args:
            path_type: "image", "video", or "save"
            
        Returns:
            Directory path or None
        """
        return self.get("paths", f"last_{path_type}_dir")
    
    def add_recent_file(self, filepath: str, max_recent: int = 10):
        """
        Add file to recent files list.
        
        Args:
            filepath: File path to add
            max_recent: Maximum number of recent files to keep
        """
        recent = self.get("recent_files", default=[])
        
        # Remove if already exists
        if filepath in recent:
            recent.remove(filepath)
        
        # Add to front
        recent.insert(0, filepath)
        
        # Trim to max
        recent = recent[:max_recent]
        
        self.set("recent_files", value=recent)
    
    def get_recent_files(self, max_count: int = 10) -> list:
        """
        Get list of recent files that still exist.
        
        Args:
            max_count: Maximum number to return
            
        Returns:
            List of file paths
        """
        recent = self.get("recent_files", default=[])
        # Filter out files that no longer exist
        existing = [f for f in recent if os.path.exists(f)]
        return existing[:max_count]
    
    def clear_recent_files(self):
        """Clear all recent files."""
        self.set("recent_files", value=[])

