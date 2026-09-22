"""Processing job configuration, validation, and input mode detection.

GUI preferences remain in config_manager; this module owns JSON job settings.
It has no dependency on the CLI presentation or video execution layers.
"""

import json
from pathlib import Path
from typing import Any, Dict

from dithering_lib import DitherMode, PixelizeMethod, PaletteSource
from utils import PaletteManager

# ==================== Config Schema & Validation ====================

# Valid values for config fields (derived from enums for consistency)
VALID_MODES = ["image", "video", "folder"]
VALID_PIXELIZATION_METHODS = [method.value for method in PixelizeMethod]
VALID_PALETTE_SOURCES = [source.value for source in PaletteSource]
VALID_DITHER_MODES = [mode.value for mode in DitherMode]


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_config(config: Dict[str, Any], config_path: Path, skip_input_check: bool = False) -> Dict[str, Any]:
    """
    Validate configuration and return normalized config.
    
    Args:
        config: Raw config dictionary
        config_path: Path to config file (for resolving relative paths)
        skip_input_check: If True, skip checking if input file exists (for override mode)
        
    Returns:
        Validated and normalized config
        
    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []
    
    # Required fields (but allow dummy values if we're going to override)
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
    
    # Check if input exists (skip if we're going to override it)
    if not skip_input_check and not Path(config["input"]).exists():
        raise ConfigValidationError(f"Input file/directory not found: {config['input']}")
    
    # Set defaults for optional fields
    config.setdefault("mode", None)  # Will be auto-detected
    config.setdefault("pixelization", {"enabled": False})
    config.setdefault("dithering", {"enabled": True, "mode": DitherMode.BAYER.value, "parameters": {}})
    config.setdefault("palette", {"source": PaletteSource.MEDIAN_CUT.value, "num_colors": 16, "use_gamma": False})
    config.setdefault("final_resize", {"enabled": False, "multiplier": 2})
    
    # Ensure nested defaults
    config["pixelization"].setdefault("enabled", False)
    config["pixelization"].setdefault("method", PixelizeMethod.REGULAR.value)
    config["pixelization"].setdefault("max_size", 128)
    
    config["dithering"].setdefault("enabled", True)
    config["dithering"].setdefault("mode", "bayer")
    config["dithering"].setdefault("parameters", {})
    
    config["palette"].setdefault("source", PaletteSource.MEDIAN_CUT.value)
    config["palette"].setdefault("num_colors", 16)
    config["palette"].setdefault("use_gamma", False)
    
    config["final_resize"].setdefault("enabled", False)
    config["final_resize"].setdefault("multiplier", 2)
    
    return config


def load_config(config_path: Path, skip_input_check: bool = False) -> Dict[str, Any]:
    """
    Load and validate configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        skip_input_check: If True, skip checking if input file exists (for override mode)
        
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
    return validate_config(config, config_path, skip_input_check=skip_input_check)


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


