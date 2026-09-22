"""Processing job configuration, validation, and input mode detection.

GUI preferences remain in config_manager; this module owns JSON job settings.
It has no dependency on the CLI presentation or video execution layers.
"""

import json
from copy import deepcopy
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


def _normalize_integer(section, key, label, minimum, errors):
    value = section[key]
    try:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError
        normalized = int(value)
        if normalized < minimum:
            errors.append(f"'{label}' must be at least {minimum}")
        else:
            section[key] = normalized
    except (ValueError, TypeError):
        errors.append(f"'{label}' must be an integer")


def validate_config(config: Dict[str, Any], config_path: Path, skip_input_check: bool = False) -> Dict[str, Any]:
    """Return a normalized job without modifying the caller's configuration."""
    if not isinstance(config, dict):
        raise ConfigValidationError("Configuration must be an object/dictionary")
    config = deepcopy(config)
    errors = []
    defaults = {
        "pixelization": {"enabled": False, "method": PixelizeMethod.REGULAR.value, "max_size": 128},
        "dithering": {"enabled": True, "mode": DitherMode.BAYER.value, "parameters": {}},
        "palette": {"source": PaletteSource.MEDIAN_CUT.value, "num_colors": 16, "use_gamma": False},
        "final_resize": {"enabled": False, "multiplier": 2},
    }
    for name, values in defaults.items():
        if name not in config:
            config[name] = deepcopy(values)
        elif not isinstance(config[name], dict):
            errors.append(f"'{name}' must be an object/dictionary")
        else:
            for key, value in values.items():
                config[name].setdefault(key, deepcopy(value))
    if errors:
        raise ConfigValidationError("Configuration validation failed: " + "; ".join(errors))

    for name in ('input', 'output'):
        if skip_input_check:
            # Both paths will be replaced by the CLI input override.
            config.setdefault(name, '.')
        if name not in config:
            errors.append(f"Missing required field: '{name}'")
        elif not isinstance(config[name], str) or not config[name].strip():
            errors.append(f"'{name}' must be a nonempty path string")

    config.setdefault('mode', None)
    if config['mode'] is not None and config['mode'] not in VALID_MODES:
        errors.append(f"Invalid mode: {config['mode']!r}. Must be one of: {VALID_MODES}")
    pix, dith, pal, resize = (config[name] for name in defaults)
    if pix['method'] not in VALID_PIXELIZATION_METHODS:
        errors.append(f"Invalid pixelization method: {pix['method']!r}")
    if dith['mode'] not in VALID_DITHER_MODES:
        errors.append(f"Invalid dither mode: {dith['mode']!r}")
    if not isinstance(dith['parameters'], dict):
        errors.append("'dithering.parameters' must be an object/dictionary")
    for name in ('pixelization', 'dithering', 'final_resize'):
        if not isinstance(config[name]['enabled'], bool):
            errors.append(f"'{name}.enabled' must be a boolean")
    if not isinstance(pal['use_gamma'], bool):
        errors.append("'palette.use_gamma' must be a boolean")

    _normalize_integer(pix, 'max_size', 'pixelization.max_size', 2, errors)
    _normalize_integer(pal, 'num_colors', 'palette.num_colors', 1, errors)
    _normalize_integer(resize, 'multiplier', 'final_resize.multiplier', 1, errors)

    source = pal['source']
    if not isinstance(source, str) or not source:
        errors.append("'palette.source' must be a nonempty string")
    elif source not in VALID_PALETTE_SOURCES:
        if source.startswith(('file:', 'custom:')):
            if not source.split(':', 1)[1].strip():
                errors.append("'palette.source' requires a path or palette name after ':'")
        else:
            palette_names = [p['name'] for p in PaletteManager().palettes]
            if source not in palette_names:
                errors.append(f"Invalid palette source: {source!r}")
    if errors:
        raise ConfigValidationError("Configuration validation failed:\n" +
                                    "\n".join(f"  - {error}" for error in errors))

    for name in ('input', 'output'):
        path = Path(config[name])
        if not path.is_absolute():
            path = Path(config_path).parent / path
        config[name] = str(path.resolve())
    if not skip_input_check and not Path(config['input']).exists():
        raise ConfigValidationError(f"Input file/directory not found: {config['input']}")
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


