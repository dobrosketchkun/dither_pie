"""Shared neural and nearest-neighbor pixelization for images and video frames."""

import os
import tempfile
from typing import Optional, Tuple
from PIL import Image

__all__ = ['NeuralPixelizer', 'pixelize_regular']


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
        import torch
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
            
            # Clear PyTorch cache after inference to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
    
    @staticmethod
    def _compute_even_dimensions(orig_w: int, orig_h: int, max_size: int) -> Tuple[int, int]:
        """Compute even dimensions for video codec compatibility."""
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 2:
            raise ValueError("max_size must be an integer of at least 2")
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Image dimensions must be positive")
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
    # NEAREST never blends, so alpha stays exactly as it was - preserve it.
    # Anything without an alpha channel still comes back as RGB, unchanged.
    if resized.mode == 'RGBA':
        return resized
    return resized.convert('RGB')

