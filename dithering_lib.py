"""
A Python library providing dithering strategies, color-reduction, gamma transforms,
and multiple dithering methods, now including a "hybrid dithering" approach.
Use this as a standalone library or import it from your application.
"""

import math
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from PIL import Image
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import pywt  # For wavelet-based dithering if needed
import heapq

# -------------------- Enumerations --------------------

class DitherMode(Enum):
    NONE = "none"
    BAYER = "bayer"
    ERROR_DIFFUSION = "error_diffusion"
    RIEMERSMA = "riemersma"
    BLUE_NOISE = "blue_noise"
    INTERLEAVED_GRADIENT_NOISE = "IGN"
    POLKA_DOT = "polka_dot"
    WAVELET = "wavelet"
    ADAPTIVE_VARIANCE = "adaptive_variance"
    PERCEPTUAL = "perceptual"
    HYBRID = "hybrid"
    HALFTONE = "halftone"
    OSTROMOUKHOV = "ostromoukhov"


# -------------------- Base Classes for Dithering Strategies --------------------

class BaseDitherStrategy:
    """
    Base class for dithering strategies.
    Each strategy must implement a .dither(pixels, palette_arr, image_size) method
    that returns an array of shape (N,3), the same shape as 'pixels'.
    """
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        raise NotImplementedError


class NoDitherStrategy(BaseDitherStrategy):
    """
    No dithering at all; simply assign each pixel to its nearest palette color.
    """
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        tree = KDTree(palette_arr)
        _, idx = tree.query(pixels, k=1, workers=-1)
        return palette_arr[idx,:]


# -------------------- Matrix-based Dithering (Bayer/Blue-Noise/Polka-Dot) --------------------

class MatrixDitherStrategy(BaseDitherStrategy):
    """
    Matrix-based dithering (Bayer, Blue-noise, polka-dot threshold, etc.).
    We compare the ratio of distances (nearest vs second-nearest color) to
    a threshold from a matrix, deciding which color to pick.
    """
    def __init__(self, threshold_matrix: np.ndarray):
        self.threshold_matrix = threshold_matrix

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h, w = image_size
        tree = KDTree(palette_arr)
        # nearest 2 palette colors
        distances, indices = tree.query(pixels, k=2, workers=-1)
        dist_sq = distances**2
        dist_nearest = dist_sq[:,0]
        dist_second = dist_sq[:,1]
        total_dist = dist_nearest + dist_second
        factor = np.where(total_dist==0, 0.0, dist_nearest / total_dist)

        # tile threshold matrix
        th_h, th_w = self.threshold_matrix.shape
        tiled = np.tile(self.threshold_matrix,
                        ((h + th_h - 1)//th_h, (w + th_w - 1)//th_w))
        tiled = tiled[:h,:w]
        flat_thresh = tiled.flatten()

        idx_nearest = indices[:,0]
        idx_second = indices[:,1]
        use_nearest = (factor <= flat_thresh)
        final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)
        return palette_arr[final_indices,:]


def generate_blue_noise(size: int=64, seed: int=42)->np.ndarray:
    """
    Generate a blue-noise threshold matrix [0..1], naive O(n^3) method.
    """
    rng = np.random.RandomState(seed)
    BN = np.zeros((size,size), dtype=np.float32)
    min_dist = np.full((size,size), np.inf, dtype=np.float32)
    coords = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(coords)
    for i in range(size*size):
        best = max(coords, key=lambda xy: min_dist[xy[0], xy[1]])
        BN[best[0], best[1]] = i / float(size*size - 1 + 1e-9)
        coords.remove(best)
        br, bc = best
        for (rr, cc) in coords:
            d2 = (rr - br)**2 + (cc - bc)**2
            if d2 < min_dist[rr, cc]:
                min_dist[rr, cc] = d2
    return BN





class BayerDitherStrategy(MatrixDitherStrategy):
    """
    Bayer ordered dithering with configurable matrix size.
    Uses threshold matrices to create characteristic crosshatch patterns.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'size': {
                'type': 'choice',
                'default': '4x4',
                'choices': ['2x2', '4x4', '8x8', '16x16'],
                'label': 'Matrix Size',
                'description': 'Size of the Bayer threshold matrix (larger = finer patterns)'
            }
        }
    
    def __init__(self, size: str = '4x4'):
        """
        Initialize Bayer dithering with specified matrix size.
        
        Args:
            size: Matrix size - '2x2', '4x4', '8x8', or '16x16'
        """
        self.size = size
        
        # Select the appropriate Bayer matrix
        if size == '2x2':
            matrix = DitherUtils.BAYER2x2
        elif size == '4x4':
            matrix = DitherUtils.BAYER4x4
        elif size == '8x8':
            matrix = DitherUtils.BAYER8x8
        elif size == '16x16':
            matrix = DitherUtils.BAYER16x16
        else:
            matrix = DitherUtils.BAYER4x4  # Default fallback
        
        super().__init__(matrix)
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'size': self.size
        }


class BlueNoiseDitherStrategy(MatrixDitherStrategy):
    """
    Blue noise dithering for high-quality spatial distribution.
    Generates a noise pattern that minimizes low-frequency artifacts.
    """
    
    # In-memory cache for generated blue noise matrices (does not persist between app runs)
    _cache = {}
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'size': {
                'type': 'int',
                'default': 64,
                'min': 32,
                'max': 128,
                'label': 'Matrix Size',
                'description': 'Size of the blue noise matrix (larger = more detail but slower)'
            },
            'seed': {
                'type': 'int',
                'default': 42,
                'min': 0,
                'max': 9999,
                'label': 'Random Seed',
                'description': 'Seed for noise generation (different seeds = different patterns)'
            }
        }
    
    def __init__(self, size: int = 64, seed: int = 42):
        self.size = size
        self.seed = seed
        
        # Check cache first
        cache_key = (size, seed)
        if cache_key in BlueNoiseDitherStrategy._cache:
            bn = BlueNoiseDitherStrategy._cache[cache_key]
        else:
            # Generate and cache
            bn = generate_blue_noise(size, seed)
            BlueNoiseDitherStrategy._cache[cache_key] = bn
        
        super().__init__(bn)
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'size': self.size,
            'seed': self.seed
        }


class InterleavedGradientNoiseDitherStrategy(BaseDitherStrategy):
    """
    Interleaved Gradient Noise (IGN) dithering.
    Generates deterministic per-pixel thresholds using pixel coordinates.
    """

    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'scale': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'label': 'Scale',
                'description': 'Noise frequency (lower = larger pattern, higher = finer grain)'
            },
            'seed': {
                'type': 'int',
                'default': 0,
                'min': 0,
                'max': 9999,
                'label': 'Seed',
                'description': 'Deterministic offset to shift the pattern'
            }
        }

    def __init__(self, scale: float = 1.0, seed: int = 0):
        self.scale = float(scale)
        self.seed = int(seed)

    @staticmethod
    def _fract(value: np.ndarray) -> np.ndarray:
        return value - np.floor(value)

    def _generate_thresholds(self, image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # Classic IGN hash-style pattern in [0, 1)
        xv = (xv + self.seed * 0.37) * self.scale
        yv = (yv + self.seed * 0.73) * self.scale
        t = self._fract(xv * 0.06711056 + yv * 0.00583715)
        return self._fract(t * 52.9829189)

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        tree = KDTree(palette_arr)

        distances, indices = tree.query(pixels, k=2, workers=-1)
        dist_sq = distances**2
        dist_nearest = dist_sq[:, 0]
        dist_second = dist_sq[:, 1]
        total_dist = dist_nearest + dist_second
        factor = np.where(total_dist == 0, 0.0, dist_nearest / total_dist)

        thresholds = self._generate_thresholds((h, w)).flatten()
        idx_nearest = indices[:, 0]
        idx_second = indices[:, 1]
        use_nearest = (factor <= thresholds)
        final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)
        return palette_arr[final_indices, :]


# -------------------- Unified Error Diffusion Strategy --------------------

class ErrorDiffusionDitherStrategy(BaseDitherStrategy):
    """
    Unified error diffusion strategy supporting multiple classic algorithms.
    All variants use fixed-weight error distribution to neighboring pixels.
    
    Variants:
    - Floyd-Steinberg: 4 neighbors, classic and fast
    - JJN: 12 neighbors, smooth gradients
    - Stucki: 12 neighbors, photographic quality
    - Burkes: 7 neighbors, balanced speed/quality
    - Atkinson: 6 neighbors, classic Mac look (loses 25% error)
    - Sierra: 10 neighbors, high quality
    - Sierra Two-Row: 8 neighbors, balanced
    - Sierra Lite: 4 neighbors, fast
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'variant': {
                'type': 'choice',
                'default': 'atkinson',
                'choices': ['floyd_steinberg', 'jjn', 'stucki', 'burkes', 'atkinson', 
                           'sierra', 'sierra_two_row', 'sierra_lite'],
                'label': 'Algorithm',
                'description': 'Error diffusion algorithm variant'
            },
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, variant: str = 'atkinson', serpentine: str = 'false'):
        """
        Initialize error diffusion with specified variant.
        
        Args:
            variant: Algorithm variant to use
            serpentine: Whether to use serpentine scanning
        """
        self.variant = variant
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'variant': self.variant,
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error based on variant
                if self.variant == 'floyd_steinberg':
                    self._distribute_floyd_steinberg(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'jjn':
                    self._distribute_jjn(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'stucki':
                    self._distribute_stucki(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'burkes':
                    self._distribute_burkes(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'atkinson':
                    self._distribute_atkinson(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'sierra':
                    self._distribute_sierra_full(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'sierra_two_row':
                    self._distribute_sierra_two_row(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'sierra_lite':
                    self._distribute_sierra_lite(work_2d, x, y, w, h, err, x_direction)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))
    
    def _distribute_floyd_steinberg(self, work_2d, x, y, w, h, err, x_direction):
        """Floyd-Steinberg - 4 neighbors, divisor = 16"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (7.0 / 16.0)
        
        if y + 1 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (3.0 / 16.0)
            
            work_2d[y + 1, x] += err * (5.0 / 16.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 16.0)
    
    def _distribute_jjn(self, work_2d, x, y, w, h, err, x_direction):
        """JJN - 12 neighbors, divisor = 48"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (7.0 / 48.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (5.0 / 48.0)
        
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (3.0 / 48.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (5.0 / 48.0)
            
            work_2d[y + 1, x] += err * (7.0 / 48.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (5.0 / 48.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (3.0 / 48.0)
        
        if y + 2 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (1.0 / 48.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (3.0 / 48.0)
            
            work_2d[y + 2, x] += err * (5.0 / 48.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (3.0 / 48.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (1.0 / 48.0)
    
    def _distribute_stucki(self, work_2d, x, y, w, h, err, x_direction):
        """Stucki - 12 neighbors, divisor = 42"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (8.0 / 42.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (4.0 / 42.0)
        
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 42.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 42.0)
            
            work_2d[y + 1, x] += err * (8.0 / 42.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 42.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 42.0)
        
        if y + 2 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (1.0 / 42.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 42.0)
            
            work_2d[y + 2, x] += err * (4.0 / 42.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 42.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (1.0 / 42.0)
    
    def _distribute_burkes(self, work_2d, x, y, w, h, err, x_direction):
        """Burkes - 7 neighbors, divisor = 32"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (8.0 / 32.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (4.0 / 32.0)
        
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            work_2d[y + 1, x] += err * (8.0 / 32.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
    
    def _distribute_atkinson(self, work_2d, x, y, w, h, err, x_direction):
        """Atkinson - 6 neighbors, divisor = 8 (only 6/8 distributed)"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (1.0 / 8.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (1.0 / 8.0)
        
        if y + 1 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 8.0)
            
            work_2d[y + 1, x] += err * (1.0 / 8.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 8.0)
        
        if y + 2 < h:
            work_2d[y + 2, x] += err * (1.0 / 8.0)
    
    def _distribute_sierra_full(self, work_2d, x, y, w, h, err, x_direction):
        """Sierra (full) - 10 neighbors, divisor = 32"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (5.0 / 32.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (3.0 / 32.0)
        
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            work_2d[y + 1, x] += err * (5.0 / 32.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
        
        if y + 2 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 32.0)
            
            work_2d[y + 2, x] += err * (3.0 / 32.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 32.0)
    
    def _distribute_sierra_two_row(self, work_2d, x, y, w, h, err, x_direction):
        """Two-Row Sierra - 8 neighbors, divisor = 16"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (4.0 / 16.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (3.0 / 16.0)
        
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 16.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 16.0)
            
            work_2d[y + 1, x] += err * (3.0 / 16.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 16.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 16.0)
    
    def _distribute_sierra_lite(self, work_2d, x, y, w, h, err, x_direction):
        """Sierra Lite - 4 neighbors, divisor = 4"""
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (2.0 / 4.0)
        
        if y + 1 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 4.0)
            
            work_2d[y + 1, x] += err * (1.0 / 4.0)


class PolkaDotDitherStrategy(BaseDitherStrategy):
    """
    Polka-dot dithering using circular threshold patterns.
    Creates a repeating pattern of circular dots for a retro printing effect.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'tile_size': {
                'type': 'int',
                'default': 8,
                'min': 4,
                'max': 32,
                'label': 'Tile Size',
                'description': 'Size of the repeating dot pattern'
            },
            'gamma': {
                'type': 'float',
                'default': 1.5,
                'min': 0.5,
                'max': 3.0,
                'step': 0.1,
                'label': 'Gamma',
                'description': 'Controls dot shape curve (higher = sharper edges)'
            }
        }
    
    def __init__(self, tile_size: int = 8, gamma: float = 1.5):
        self.tile_size = tile_size
        self.gamma = gamma
        self.threshold_matrix = self._generate_polka_dot_matrix(tile_size, gamma)
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'tile_size': self.tile_size,
            'gamma': self.gamma
        }

    def _generate_polka_dot_matrix(self, tile_size: int, gamma: float) -> np.ndarray:
        x = np.arange(tile_size)
        y = np.arange(tile_size)
        xv, yv = np.meshgrid(x, y)
        cx = (tile_size - 1) / 2
        cy = (tile_size - 1) / 2
        dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = dist / (max_dist + 1e-9)
        thresh = 1.0 - (norm_dist**gamma)
        return np.clip(thresh, 0, 1).astype(np.float32)

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h, w = image_size
        tree = KDTree(palette_arr)
        distances, indices = tree.query(pixels, k=2, workers=-1)
        dist_sq = distances**2
        dist_nearest = dist_sq[:,0]
        dist_second = dist_sq[:,1]
        total_dist = dist_nearest + dist_second
        factor = np.where(total_dist==0, 0.0, dist_nearest/total_dist)

        th_h, th_w = self.threshold_matrix.shape
        tiled = np.tile(self.threshold_matrix,
                        ((h + th_h - 1)//th_h, (w + th_w - 1)//th_w))
        tiled = tiled[:h,:w]
        flat_thresh = tiled.flatten()

        idx_nearest = indices[:,0]
        idx_second = indices[:,1]
        use_nearest = (factor <= flat_thresh)
        final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)
        return palette_arr[final_indices,:]


# -------------------- Riemersma (Hilbert-based Error Diffusion) --------------------

def _hilbert_order(n:int)->np.ndarray:
    """
    Generate Hilbert curve ordering for n x n (n=2^k). Return shape (n*n,2).
    """
    order_map = np.zeros((n,n), dtype=np.int32)
    def hilbert_xy(index, order_bits):
        s=1
        x=y=0
        t=index
        for lvl in range(order_bits):
            rx = 1 & (t//2)
            ry = 1 & (t^rx)
            if ry==0:
                if rx==1:
                    x=s-1 - x
                    y=s-1 - y
                x,y = y,x
            x += s*rx
            y += s*ry
            t//=4
            s<<=1
        return x,y

    order_bits = int(math.log2(n))
    N = n*n
    for i in range(N):
        xx, yy = hilbert_xy(i, order_bits)
        order_map[yy, xx] = i
    coords = np.empty((N,2), dtype=np.int32)
    for rr in range(n):
        for cc in range(n):
            idx = order_map[rr,cc]
            coords[idx] = [rr, cc]
    return coords

def _next_power_of_two(x:int)->int:
    return 2**int(math.ceil(math.log2(x))) if x>0 else 1

class RiemersmaDitherStrategy(BaseDitherStrategy):
    """
    Hilbert-based error diffusion. Single pass with Floyd–Steinberg-like weights.
    """
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h,w = image_size
        pix_2d = pixels.reshape((h,w,3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        dim = _next_power_of_two(max(h,w))
        path = _hilbert_order(dim)
        fs_weights = [7/16, 1/16, 5/16, 3/16]
        N = len(path)
        for i in range(N):
            rr, cc = path[i]
            if rr>=h or cc>=w:
                continue
            old_val = pix_2d[rr, cc].copy()
            _, idx = tree.query(old_val, k=1)
            chosen = palette_arr[idx]
            pix_2d[rr, cc] = chosen
            err = old_val - chosen
            for off, weight in enumerate(fs_weights, start=1):
                j = i+off
                if j<N:
                    rr2, cc2 = path[j]
                    if rr2<h and cc2<w:
                        pix_2d[rr2, cc2] += err*weight
                        np.clip(pix_2d[rr2, cc2], 0, 255, out=pix_2d[rr2, cc2])
        return pix_2d.reshape((-1,3))


# -------------------- Wavelet-based Dithering --------------------

class WaveletDitherStrategy(BaseDitherStrategy):
    """
    A multi-scale approach that does wavelet decomposition, subband quantization,
    then inverse wavelet, and finally a 2-color nearest selection from the palette.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'wavelet': {
                'type': 'choice',
                'default': 'haar',
                'choices': ['haar', 'db1', 'db2', 'db4', 'sym2', 'sym4', 'coif1', 'bior1.3', 'bior2.2'],
                'label': 'Wavelet Type',
                'description': 'Type of wavelet basis function (haar = simplest, db = Daubechies, sym = Symlets)'
            },
            'subband_quant': {
                'type': 'int',
                'default': 8,
                'min': 2,
                'max': 32,
                'label': 'Subband Quantization',
                'description': 'Number of quantization levels for wavelet subbands (higher = smoother)'
            }
        }
    
    def __init__(self, wavelet: str = 'haar', subband_quant: int = 8):
        self.wavelet = wavelet
        self.subband_quant = subband_quant
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'wavelet': self.wavelet,
            'subband_quant': self.subband_quant
        }

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h,w = image_size
        pix_3d = pixels.reshape((h,w,3)).copy()
        result_3d = np.zeros_like(pix_3d)

        for ch in range(3):
            channel_data = pix_3d[:,:,ch]
            coeffs2 = pywt.dwt2(channel_data, self.wavelet)
            cA,(cH,cV,cD) = coeffs2
            cA_d = self._quant_subband(cA)
            cH_d = self._quant_subband(cH)
            cV_d = self._quant_subband(cV)
            cD_d = self._quant_subband(cD)
            new_coeffs2 = (cA_d,(cH_d,cV_d,cD_d))
            rec_channel = pywt.idwt2(new_coeffs2, self.wavelet)
            rec_channel = rec_channel[:h,:w]
            np.clip(rec_channel, 0, 255, out=rec_channel)
            result_3d[:,:,ch] = rec_channel

        # final pass: pick from palette
        flat_final = result_3d.reshape((-1,3))
        tree = KDTree(palette_arr)
        distances, indices = tree.query(flat_final, k=2, workers=-1)
        dist_sq = distances**2
        dn = dist_sq[:,0]
        ds = dist_sq[:,1]
        tot = dn+ds
        factor = np.where(tot==0,0.0,dn/tot)
        rand_thr = np.random.rand(len(flat_final))
        idx_nearest = indices[:,0]
        idx_second = indices[:,1]
        use_nearest = (factor <= rand_thr)
        final_idx = np.where(use_nearest, idx_nearest, idx_second)
        return palette_arr[final_idx,:]

    def _quant_subband(self, subband: np.ndarray)->np.ndarray:
        mn = subband.min()
        mx = subband.max()
        if mx==mn:
            return subband.astype(np.float32)
        noise = np.random.rand(*subband.shape).astype(np.float32)
        scale = mx - mn
        norm = (subband - mn)/(scale+1e-9)
        q = norm * self.subband_quant
        q += noise
        q = np.floor(q)
        np.clip(q,0,self.subband_quant-1,out=q)
        qn = q/(self.subband_quant-1+1e-9)
        out = qn*scale + mn
        return out.astype(np.float32)


# -------------------- Adaptive Variance Dithering --------------------

class AdaptiveVarianceDitherStrategy(BaseDitherStrategy):
    """
    Demonstrates a simple adaptive dithering that checks local variance:
      - If the local variance is below a threshold => skip error diffusion (just pick nearest color).
      - If it's above the threshold => do a small Floyd–Steinberg distribution to neighbors.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'var_threshold': {
                'type': 'float',
                'default': 300.0,
                'min': 0.0,
                'max': 1000.0,
                'step': 10.0,
                'label': 'Variance Threshold',
                'description': 'Threshold for local variance to trigger error diffusion (higher = less diffusion)'
            },
            'window_radius': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 5,
                'label': 'Window Radius',
                'description': 'Radius of window for computing local variance (larger = smoother adaptation)'
            }
        }
    
    def __init__(self, var_threshold: float = 300.0, window_radius: int = 1):
        self.var_threshold = var_threshold
        self.window_radius = window_radius
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'var_threshold': self.var_threshold,
            'window_radius': self.window_radius
        }

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h, w = image_size

        # We'll compute a local variance map from grayscale.
        pix_2d = pixels.reshape((h,w,3))
        gray_2d = 0.299*pix_2d[:,:,0] + 0.587*pix_2d[:,:,1] + 0.114*pix_2d[:,:,2]

        var_map = self._compute_variance_map(gray_2d)

        # We'll do a row-major error diffusion pass.
        work_2d = pix_2d.astype(np.float32).copy()
        tree = KDTree(palette_arr)

        for y in range(h):
            for x in range(w):
                old_val = work_2d[y, x].copy()
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                err = old_val - chosen

                # if local variance is high => do small Floyd–Steinberg distribution
                if var_map[y, x] >= self.var_threshold:
                    if x+1 < w:
                        work_2d[y, x+1] += err*(7/16)
                    if (y+1 < h) and (x>0):
                        work_2d[y+1, x-1] += err*(3/16)
                    if y+1 < h:
                        work_2d[y+1, x]   += err*(5/16)
                    if (y+1 < h) and (x+1 < w):
                        work_2d[y+1, x+1] += err*(1/16)

        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1,3))

    def _compute_variance_map(self, gray_2d: np.ndarray)->np.ndarray:
        h, w = gray_2d.shape
        var_map = np.zeros((h,w), dtype=np.float32)
        wr = self.window_radius
        ws = 2*wr + 1

        # naive approach (O(N * ws^2))
        for y in range(h):
            y1 = max(0, y-wr)
            y2 = min(h, y+wr+1)
            for x in range(w):
                x1 = max(0, x-wr)
                x2 = min(w, x+wr+1)
                region = gray_2d[y1:y2, x1:x2]
                variance = region.var()
                var_map[y,x] = variance
        return var_map


# -------------------- Perceptual Dithering --------------------

class PerceptualDitherStrategy(BaseDitherStrategy):
    """
    A simple demonstration of "perceptual dithering," combining:
      - Row-major error diffusion (like Floyd–Steinberg).
      - A local luminance-based scaling of the error distribution.
    """
    def __init__(self, base_weights=None):
        if base_weights is None:
            self.base_weights = [(1,0,7/16), (-1,1,3/16), (0,1,5/16), (1,1,1/16)]
        else:
            self.base_weights = base_weights

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h,w,3)).astype(np.float32).copy()
        gray_2d = 0.299*work_2d[:,:,0] + 0.587*work_2d[:,:,1] + 0.114*work_2d[:,:,2]
        tree = KDTree(palette_arr)

        for y in range(h):
            for x in range(w):
                old_val = work_2d[y, x].copy()
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                err = old_val - chosen

                lum = gray_2d[y, x]
                sensitivity = 0.5 + 0.5*(lum/255.0)  # simple example
                for (dx, dy, wgt) in self.base_weights:
                    nx = x + dx
                    ny = y + dy
                    if (0 <= nx < w) and (0 <= ny < h):
                        work_2d[ny, nx] += err*(wgt*sensitivity)

        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1,3))


# -------------------- NEW: Hybrid Dithering --------------------

class HybridDitherStrategy(BaseDitherStrategy):
    """
    Demonstrates a "hybrid" approach: we do a row-major error diffusion,
    but we only fully diffuse the luminance portion of the error, while
    reducing (or omitting) the color portion.

    For each pixel:
      1) find nearest palette color
      2) error = (R_old - R_chosen, G_old - G_chosen, B_old - B_chosen)
      3) separate error into luminance vs color:
         lum_err_val = 0.299*errR + 0.587*errG + 0.114*errB
         err_lum = lum_err_val * [0.299, 0.587, 0.114]
         err_col = err - err_lum
      4) final error = lum_factor*err_lum + col_factor*err_col
      5) distribute final error to neighbors
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'lum_factor': {
                'type': 'float',
                'default': 1.0,
                'min': 0.0,
                'max': 2.0,
                'step': 0.1,
                'label': 'Luminance Factor',
                'description': 'Strength of luminance error diffusion (1.0 = full, 0.0 = none)'
            },
            'col_factor': {
                'type': 'float',
                'default': 0.2,
                'min': 0.0,
                'max': 2.0,
                'step': 0.1,
                'label': 'Color Factor',
                'description': 'Strength of color error diffusion (lower = less color noise)'
            }
        }

    def __init__(self, lum_factor: float = 1.0, col_factor: float = 0.2):
        """
        lum_factor: how strongly to diffuse the luminance portion
        col_factor: how strongly to diffuse the color portion
        For instance, (lum_factor=1.0, col_factor=0.2) means we
        preserve grayscale detail strongly while limiting color noise.
        """
        self.lum_factor = lum_factor
        self.col_factor = col_factor
        # We'll use standard Floyd–Steinberg offsets & weights
        self.fs_offsets = [(1,0,7/16), ( -1,1,3/16 ), (0,1,5/16), (1,1,1/16)]
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'lum_factor': self.lum_factor,
            'col_factor': self.col_factor
        }

    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int,int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h,w,3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)

        for y in range(h):
            for x in range(w):
                old_val = work_2d[y,x].copy()
                # nearest color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y,x] = chosen

                # compute error
                err = old_val - chosen
                # separate into luminance & color portion
                lum_err_val = 0.299*err[0] + 0.587*err[1] + 0.114*err[2]
                err_lum = np.array([0.299, 0.587, 0.114], dtype=np.float32)*lum_err_val
                err_col = err - err_lum

                # scale them
                final_err = self.lum_factor*err_lum + self.col_factor*err_col

                # distribute
                for (dx, dy, weight) in self.fs_offsets:
                    nx = x+dx
                    ny = y+dy
                    if 0<=nx<w and 0<=ny<h:
                        work_2d[ny,nx] += final_err*weight

        np.clip(work_2d,0,255,out=work_2d)
        return work_2d.reshape((-1,3))


# -------------------- Floyd-Steinberg Error Diffusion --------------------

class FloydSteinbergDitherStrategy(BaseDitherStrategy):
    """
    Classic Floyd-Steinberg error diffusion algorithm.
    Uses fixed weights to distribute quantization error to neighboring pixels.
    
    Error distribution pattern:
           [ ] [X] [ ]
           [3] [5] [1]  (divided by 16, with 7 going right from X)
    
    This is the most well-known dithering algorithm and produces good results
    for most images.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize Floyd-Steinberg error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning (alternating row directions)
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error using Floyd-Steinberg weights
                # Pattern:      [X] 7→
                #           3↙  5↓  1↘
                
                # Right pixel (7/16)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (7.0 / 16.0)
                
                # Bottom row
                if y + 1 < h:
                    # Bottom-left pixel (3/16)
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (3.0 / 16.0)
                    
                    # Bottom pixel (5/16)
                    work_2d[y + 1, x] += err * (5.0 / 16.0)
                    
                    # Bottom-right pixel (1/16)
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (1.0 / 16.0)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Jarvis, Judice & Ninke (JJN) Error Diffusion --------------------

class JJNDitherStrategy(BaseDitherStrategy):
    """
    Jarvis, Judice & Ninke (JJN) error diffusion algorithm.
    Uses a larger diffusion kernel than Floyd-Steinberg (12 neighbors instead of 4)
    for smoother gradients and fewer artifacts.
    
    Error distribution pattern (divided by 48):
               [ ] [ ] [X] 7  5
           3   5   7   5   3
           1   3   5   3   1
    
    This algorithm is slower than Floyd-Steinberg but produces superior quality,
    especially in smooth gradients.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize JJN error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning (alternating row directions)
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error using JJN weights (divided by 48)
                # Pattern:      [X] 7→ 5→
                #          3←  5← 7← 5← 3←
                #          1←  3← 5← 3← 1←
                
                # Current row (y)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (7.0 / 48.0)
                
                nx = x + 2 * x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (5.0 / 48.0)
                
                # Next row (y+1)
                if y + 1 < h:
                    nx = x - 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (3.0 / 48.0)
                    
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (5.0 / 48.0)
                    
                    work_2d[y + 1, x] += err * (7.0 / 48.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (5.0 / 48.0)
                    
                    nx = x + 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (3.0 / 48.0)
                
                # Two rows down (y+2)
                if y + 2 < h:
                    nx = x - 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (1.0 / 48.0)
                    
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (3.0 / 48.0)
                    
                    work_2d[y + 2, x] += err * (5.0 / 48.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (3.0 / 48.0)
                    
                    nx = x + 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (1.0 / 48.0)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Atkinson Error Diffusion --------------------

class AtkinsonDitherStrategy(BaseDitherStrategy):
    """
    Atkinson error diffusion algorithm by Bill Atkinson.
    Created for the original Apple Macintosh, known for its distinctive look.
    
    Error distribution pattern (divided by 8, but only 6/8 distributed):
           [ ] [X] 1  1
           1   1   1
               1
    
    Key characteristic: Only distributes 75% (6/8) of the error, intentionally
    losing 25%. This creates a lighter, higher-contrast appearance that's
    characteristic of classic Mac graphics.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize Atkinson error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error using Atkinson weights (divided by 8)
                # Only 6/8 of error is distributed, 2/8 is lost
                # Pattern:      [X] 1→ 1→
                #           1← 1← 1←
                #              1↓
                
                # Current row (y)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (1.0 / 8.0)
                
                nx = x + 2 * x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (1.0 / 8.0)
                
                # Next row (y+1)
                if y + 1 < h:
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (1.0 / 8.0)
                    
                    work_2d[y + 1, x] += err * (1.0 / 8.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (1.0 / 8.0)
                
                # Two rows down (y+2)
                if y + 2 < h:
                    work_2d[y + 2, x] += err * (1.0 / 8.0)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Sierra Family Error Diffusion --------------------

class SierraDitherStrategy(BaseDitherStrategy):
    """
    Sierra family of error diffusion algorithms by Frankie Sierra.
    Offers three variants with different kernel sizes and quality/speed tradeoffs.
    
    Variants:
    - Sierra (Full): 10 neighbors, highest quality, slowest
           [ ] [ ] [X] 5  3
       2   4   5   4   2
           2   3   2
    
    - Two-Row Sierra: 8 neighbors, balanced quality/speed
           [ ] [ ] [X] 4  3
       1   2   3   2   1
    
    - Sierra Lite: 4 neighbors, fastest, good quality
           [ ] [X] 2
       1   1
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'variant': {
                'type': 'choice',
                'default': 'sierra',
                'choices': ['sierra', 'two_row', 'lite'],
                'label': 'Variant',
                'description': 'Sierra variant (sierra=highest quality, lite=fastest)'
            },
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, variant: str = 'sierra', serpentine: str = 'false'):
        """
        Initialize Sierra error diffusion.
        
        Args:
            variant: 'sierra' (full), 'two_row', or 'lite'
            serpentine: Whether to use serpentine scanning
        """
        self.variant = variant
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'variant': self.variant,
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error based on variant
                if self.variant == 'sierra':
                    self._distribute_sierra_full(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'two_row':
                    self._distribute_sierra_two_row(work_2d, x, y, w, h, err, x_direction)
                elif self.variant == 'lite':
                    self._distribute_sierra_lite(work_2d, x, y, w, h, err, x_direction)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))
    
    def _distribute_sierra_full(self, work_2d, x, y, w, h, err, x_direction):
        """Sierra (full) - 10 neighbors, divisor = 32"""
        # Current row (y)
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (5.0 / 32.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (3.0 / 32.0)
        
        # Next row (y+1)
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            work_2d[y + 1, x] += err * (5.0 / 32.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (4.0 / 32.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 32.0)
        
        # Two rows down (y+2)
        if y + 2 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 32.0)
            
            work_2d[y + 2, x] += err * (3.0 / 32.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 2, nx] += err * (2.0 / 32.0)
    
    def _distribute_sierra_two_row(self, work_2d, x, y, w, h, err, x_direction):
        """Two-Row Sierra - 8 neighbors, divisor = 16"""
        # Current row (y)
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (4.0 / 16.0)
        
        nx = x + 2 * x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (3.0 / 16.0)
        
        # Next row (y+1)
        if y + 1 < h:
            nx = x - 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 16.0)
            
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 16.0)
            
            work_2d[y + 1, x] += err * (3.0 / 16.0)
            
            nx = x + x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (2.0 / 16.0)
            
            nx = x + 2 * x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 16.0)
    
    def _distribute_sierra_lite(self, work_2d, x, y, w, h, err, x_direction):
        """Sierra Lite - 4 neighbors, divisor = 4"""
        # Current row (y)
        nx = x + x_direction
        if 0 <= nx < w:
            work_2d[y, nx] += err * (2.0 / 4.0)
        
        # Next row (y+1)
        if y + 1 < h:
            nx = x - x_direction
            if 0 <= nx < w:
                work_2d[y + 1, nx] += err * (1.0 / 4.0)
            
            work_2d[y + 1, x] += err * (1.0 / 4.0)


# -------------------- Stucki Error Diffusion --------------------

class StuckiDitherStrategy(BaseDitherStrategy):
    """
    Stucki error diffusion algorithm.
    Uses a 12-neighbor kernel similar to JJN but with different weights.
    Known for producing smooth results with minimal artifacts.
    
    Error distribution pattern (divided by 42):
           [ ] [ ] [X] 8  4
       2   4   8   4   2
       1   2   4   2   1
    
    Stucki tends to produce very smooth gradients and is excellent for
    photographic images.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize Stucki error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error using Stucki weights (divided by 42)
                # Pattern:      [X] 8→ 4→
                #          2←  4← 8← 4← 2←
                #          1←  2← 4← 2← 1←
                
                # Current row (y)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (8.0 / 42.0)
                
                nx = x + 2 * x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (4.0 / 42.0)
                
                # Next row (y+1)
                if y + 1 < h:
                    nx = x - 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (2.0 / 42.0)
                    
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (4.0 / 42.0)
                    
                    work_2d[y + 1, x] += err * (8.0 / 42.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (4.0 / 42.0)
                    
                    nx = x + 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (2.0 / 42.0)
                
                # Two rows down (y+2)
                if y + 2 < h:
                    nx = x - 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (1.0 / 42.0)
                    
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (2.0 / 42.0)
                    
                    work_2d[y + 2, x] += err * (4.0 / 42.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (2.0 / 42.0)
                    
                    nx = x + 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 2, nx] += err * (1.0 / 42.0)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Burkes Error Diffusion --------------------

class BurkesDitherStrategy(BaseDitherStrategy):
    """
    Burkes error diffusion algorithm.
    Uses a 7-neighbor kernel with emphasis on horizontal and vertical neighbors.
    Known for being fast while producing good quality.
    
    Error distribution pattern (divided by 32):
           [ ] [ ] [X] 8  4
       2   4   8   4   2
    
    Burkes uses only 2 rows (vs 3 for JJN/Stucki) making it faster while
    still maintaining good quality.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize Burkes error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Distribute error using Burkes weights (divided by 32)
                # Pattern:      [X] 8→ 4→
                #          2←  4← 8← 4← 2←
                
                # Current row (y)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (8.0 / 32.0)
                
                nx = x + 2 * x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (4.0 / 32.0)
                
                # Next row (y+1)
                if y + 1 < h:
                    nx = x - 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (2.0 / 32.0)
                    
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (4.0 / 32.0)
                    
                    work_2d[y + 1, x] += err * (8.0 / 32.0)
                    
                    nx = x + x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (4.0 / 32.0)
                    
                    nx = x + 2 * x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (2.0 / 32.0)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Ostromoukhov's Variable Error Diffusion --------------------

class OstromoukhovDitherStrategy(BaseDitherStrategy):
    """
    Ostromoukhov's variable error diffusion algorithm.
    Uses adaptive coefficients based on pixel intensity for better quality
    than fixed Floyd-Steinberg weights.
    
    Reference: Victor Ostromoukhov, "A Simple and Efficient Error-Diffusion Algorithm" (2001)
    """
    
    # Ostromoukhov's coefficient table for intensities 0-255
    # Each entry is (c0, c1, c2) for [right, bottom-left, bottom] pixels
    # Normalized to sum to a power of 2 for efficiency
    COEFFS_TABLE = [
        # Generated from Ostromoukhov's paper
        (13, 0, 5), (13, 0, 5), (21, 0, 10), (7, 0, 4), (8, 0, 5), (47, 3, 28), (23, 3, 13), (15, 3, 8),
        (22, 6, 11), (43, 15, 20), (7, 3, 3), (501, 224, 211), (249, 116, 103), (165, 80, 67), (123, 62, 49), (489, 256, 191),
        (81, 44, 31), (483, 272, 181), (60, 35, 22), (53, 32, 19), (237, 148, 83), (471, 304, 161), (3, 2, 1), (481, 314, 185),
        (354, 226, 155), (1389, 866, 685), (227, 138, 125), (267, 158, 163), (327, 188, 220), (61, 34, 45), (627, 338, 505), (1227, 638, 1075),
        (20, 10, 19), (1937, 1000, 1767), (977, 520, 855), (657, 360, 551), (71, 40, 57), (2005, 1160, 1539), (337, 200, 247), (2039, 1240, 1425),
        (257, 160, 171), (691, 440, 437), (1045, 680, 627), (301, 200, 171), (177, 120, 95), (2141, 1480, 1083), (1079, 760, 513), (725, 520, 323),
        (137, 100, 57), (2209, 1640, 855), (53, 40, 19), (2243, 1720, 741), (565, 440, 171), (2325, 1840, 579), (589, 480, 131), (981, 820, 185),
        (331, 280, 51), (1413, 1220, 255), (355, 310, 57), (1485, 1320, 231), (79, 70, 11), (314, 280, 43), (1101, 1000, 123), (42, 38, 5),
        (481, 440, 53), (229, 210, 23), (1973, 1820, 191), (991, 920, 87), (497, 466, 37), (251, 236, 19), (983, 928, 69), (61, 58, 3),
        (497, 472, 29), (251, 238, 15), (983, 952, 35), (993, 968, 27), (1003, 982, 21), (1013, 992, 19), (1023, 1002, 17), (2033, 2012, 15),
        (513, 506, 5), (1021, 1010, 7), (511, 504, 5), (1021, 1014, 5), (511, 506, 3), (511, 507, 2), (1023, 1018, 3), (2047, 2042, 3),
        (511, 508, 1), (2045, 2044, 1), (1023, 1022, 1), (2047, 2046, 1), (1535, 1534, 1), (511, 511, 0), (1535, 1535, 0), (1023, 1023, 0),
        (511, 511, 0), (511, 511, 0), (1023, 1023, 0), (1535, 1535, 0), (2047, 2047, 0), (511, 511, 0), (511, 511, 0), (511, 511, 0),
        (511, 511, 0), (1023, 1023, 0), (1023, 1023, 0), (1023, 1023, 0), (1023, 1023, 0), (1535, 1535, 0), (1535, 1535, 0), (511, 511, 0),
        (1023, 1023, 0), (1535, 1535, 0), (511, 511, 0), (511, 511, 0), (1023, 1023, 0), (1535, 1535, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (2047, 2047, 0), (2047, 2047, 0),
        (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (1535, 1535, 0), (2047, 2047, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0),
        (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0),
        (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0),
        (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0),
        (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0),
        (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0),
        (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0), (1535, 1535, 0), (1023, 1023, 0), (2047, 2047, 0)
    ]
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        """
        return {
            'serpentine': {
                'type': 'choice',
                'default': 'false',
                'choices': ['true', 'false'],
                'label': 'Serpentine Scan',
                'description': 'Alternates direction each row to reduce artifacts'
            }
        }
    
    def __init__(self, serpentine: str = 'false'):
        """
        Initialize Ostromoukhov error diffusion.
        
        Args:
            serpentine: Whether to use serpentine scanning (alternating row directions)
        """
        self.serpentine = (serpentine == 'true')
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'serpentine': 'true' if self.serpentine else 'false'
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        work_2d = pixels.reshape((h, w, 3)).astype(np.float32).copy()
        tree = KDTree(palette_arr)
        
        for y in range(h):
            # Serpentine scanning: alternate row direction
            if self.serpentine and (y % 2 == 1):
                x_range = range(w - 1, -1, -1)
                x_direction = -1
            else:
                x_range = range(w)
                x_direction = 1
            
            for x in x_range:
                old_val = work_2d[y, x].copy()
                
                # Find nearest palette color
                _, idx = tree.query(old_val, k=1)
                chosen = palette_arr[idx]
                work_2d[y, x] = chosen
                
                # Calculate error
                err = old_val - chosen
                
                # Get variable coefficients based on pixel intensity (luminance)
                luminance = 0.299 * old_val[0] + 0.587 * old_val[1] + 0.114 * old_val[2]
                intensity_idx = int(np.clip(luminance, 0, 255))
                c0, c1, c2 = self.COEFFS_TABLE[intensity_idx]
                divisor = c0 + c1 + c2
                
                if divisor == 0:
                    continue
                
                # Distribute error using variable coefficients
                # Pattern:  [ ] [X] [ ]
                #           [ ] [ ] [ ]
                # becomes:  [X] c0→
                #           c1↙ c2↓
                
                # Right pixel (c0)
                nx = x + x_direction
                if 0 <= nx < w:
                    work_2d[y, nx] += err * (c0 / divisor)
                
                # Bottom row
                if y + 1 < h:
                    # Bottom-left pixel (c1)
                    nx = x - x_direction
                    if 0 <= nx < w:
                        work_2d[y + 1, nx] += err * (c1 / divisor)
                    
                    # Bottom pixel (c2)
                    work_2d[y + 1, x] += err * (c2 / divisor)
        
        np.clip(work_2d, 0, 255, out=work_2d)
        return work_2d.reshape((-1, 3))


# -------------------- Halftone Dithering --------------------

class HalftoneDitherStrategy(BaseDitherStrategy):
    """
    Newspaper-style halftone dithering using variable-sized dots on a regular grid.
    Creates the classic printed newspaper look with dots that vary in size based on brightness.
    """
    
    @staticmethod
    def get_parameter_info():
        """
        Returns metadata about configurable parameters for this dithering mode.
        This allows the GUI to build a settings dialog automatically.
        """
        return {
            'cell_size': {
                'type': 'int',
                'default': 8,
                'min': 2,
                'max': 32,
                'label': 'Cell Size',
                'description': 'Distance between dot centers (smaller = finer detail)'
            },
            'angle': {
                'type': 'float',
                'default': 45.0,
                'min': 0.0,
                'max': 90.0,
                'label': 'Screen Angle',
                'description': 'Rotation angle in degrees (45° is classic newspaper)'
            },
            'dot_gain': {
                'type': 'float',
                'default': 1.0,
                'min': 0.5,
                'max': 3.0,
                'step': 0.1,
                'label': 'Dot Gain',
                'description': 'Controls dot growth (1.0 = linear, higher = more contrast)'
            },
            'min_dot_size': {
                'type': 'float',
                'default': 0.0,
                'min': 0.0,
                'max': 0.5,
                'step': 0.05,
                'label': 'Min Dot Size',
                'description': 'Minimum dot threshold (0 = pure white possible)'
            },
            'max_dot_size': {
                'type': 'float',
                'default': 1.0,
                'min': 0.5,
                'max': 1.0,
                'step': 0.05,
                'label': 'Max Dot Size',
                'description': 'Maximum dot threshold (1.0 = pure black possible)'
            },
            'shape': {
                'type': 'choice',
                'default': 'circle',
                'choices': ['circle', 'square', 'diamond'],
                'label': 'Dot Shape',
                'description': 'Shape of halftone dots'
            },
            'sharpness': {
                'type': 'float',
                'default': 1.5,
                'min': 0.5,
                'max': 4.0,
                'step': 0.1,
                'label': 'Sharpness',
                'description': 'Edge sharpness (higher = crisper dots)'
            }
        }
    
    def __init__(self, 
                 cell_size: int = 8,
                 angle: float = 45.0,
                 dot_gain: float = 1.0,
                 min_dot_size: float = 0.0,
                 max_dot_size: float = 1.0,
                 shape: str = "circle",
                 sharpness: float = 1.5):
        """
        Initialize halftone dithering with newspaper-style parameters.
        
        Args:
            cell_size: Grid spacing between dot centers
            angle: Screen angle in degrees (45° is classic newspaper)
            dot_gain: Controls dot growth rate (higher = more contrast)
            min_dot_size: Minimum dot radius as fraction of cell_size
            max_dot_size: Maximum dot radius as fraction of cell_size
            shape: Dot shape - "circle", "square", or "diamond"
            sharpness: Edge sharpness for anti-aliasing
        """
        self.cell_size = cell_size
        self.angle = angle
        self.dot_gain = dot_gain
        self.min_dot_size = min_dot_size
        self.max_dot_size = max_dot_size
        self.shape = shape
        self.sharpness = sharpness
    
    def get_current_parameters(self):
        """Returns current parameter values."""
        return {
            'cell_size': self.cell_size,
            'angle': self.angle,
            'dot_gain': self.dot_gain,
            'min_dot_size': self.min_dot_size,
            'max_dot_size': self.max_dot_size,
            'shape': self.shape,
            'sharpness': self.sharpness
        }
    
    def dither(self, pixels: np.ndarray, palette_arr: np.ndarray,
               image_size: Tuple[int, int]) -> np.ndarray:
        h, w = image_size
        
        # Flatten pixels for processing
        flat_pixels = pixels.astype(np.float32)
        pix_2d = flat_pixels.reshape((h, w, 3))
        
        # Calculate brightness of each pixel
        gray = 0.299 * pix_2d[:,:,0] + 0.587 * pix_2d[:,:,1] + 0.114 * pix_2d[:,:,2]
        gray_norm = gray / 255.0  # Normalize to [0, 1]
        
        # Find PAPER (lightest color) in palette - this is the background
        palette_brightness = 0.299 * palette_arr[:,0] + 0.587 * palette_arr[:,1] + 0.114 * palette_arr[:,2]
        paper_idx = np.argmax(palette_brightness)
        
        # Build KDTree for palette
        tree = KDTree(palette_arr)
        
        # Generate halftone screen and get cell assignments
        halftone_screen, cell_assignments = self._generate_halftone_screen_with_cells(h, w)
        
        # OPTIMIZED: Compute average color per cell using bincount (vectorized)
        # Flatten everything for easier processing
        cell_ids_flat = cell_assignments.flatten()
        pixels_flat = pix_2d.reshape(-1, 3)
        
        # Find unique cells and their average colors
        unique_cells = np.unique(cell_ids_flat)
        num_cells = len(unique_cells)
        
        # Create a mapping from cell_id to a contiguous index for faster access (vectorized)
        # Use searchsorted for fast lookup
        cell_idx_flat = np.searchsorted(unique_cells, cell_ids_flat)
        
        # Compute sum of colors per cell (vectorized)
        cell_color_sum = np.zeros((num_cells, 3), dtype=np.float64)
        cell_count = np.bincount(cell_idx_flat, minlength=num_cells)
        
        for c in range(3):  # For each color channel
            cell_color_sum[:, c] = np.bincount(cell_idx_flat, weights=pixels_flat[:, c], minlength=num_cells)
        
        # Compute average color per cell
        cell_avg_colors = cell_color_sum / np.maximum(cell_count[:, np.newaxis], 1)
        
        # Query all cell colors at once (batch query is faster)
        _, cell_palette_indices = tree.query(cell_avg_colors, k=1)
        
        # Create a lookup array from cell_id to palette index
        cell_id_to_palette = np.zeros(unique_cells.max() + 1, dtype=np.int32)
        cell_id_to_palette[unique_cells] = cell_palette_indices.flatten()
        
        # NEWSPAPER LOGIC:
        # Start with PAPER everywhere
        # Place INK dots where darkness > threshold
        # Each dot is ONE solid color
        
        darkness = 1.0 - gray_norm  # How much ink is needed
        place_ink = darkness > halftone_screen
        
        # Start with paper everywhere
        result = np.full((h, w), paper_idx, dtype=np.int32)
        
        # Place ink dots - vectorized assignment
        # Where ink is needed, use the cell's assigned color
        result[place_ink] = cell_id_to_palette[cell_assignments[place_ink]]
        
        return palette_arr[result.flatten(), :]
    
    def _generate_halftone_screen_with_cells(self, h: int, w: int) -> tuple:
        """
        Generate halftone screen and return cell assignments.
        Each cell (dot) gets a unique ID so we can assign ONE color per dot.
        
        Returns: (screen, cell_assignments)
        """
        # Convert angle to radians
        angle_rad = np.radians(self.angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Rotate coordinates
        x_rot = x_coords * cos_a - y_coords * sin_a
        y_rot = x_coords * sin_a + y_coords * cos_a
        
        # Find grid cells - each cell gets unique ID
        cell_x = np.floor(x_rot / self.cell_size).astype(np.int32)
        cell_y = np.floor(y_rot / self.cell_size).astype(np.int32)
        
        # Create unique cell IDs
        # Offset to ensure positive IDs
        cell_x_offset = cell_x - cell_x.min()
        cell_y_offset = cell_y - cell_y.min()
        max_x = cell_x_offset.max() + 1
        cell_assignments = cell_y_offset * max_x + cell_x_offset
        
        # Calculate position within cell (fractional part)
        x_in_cell = (x_rot % self.cell_size) / self.cell_size
        y_in_cell = (y_rot % self.cell_size) / self.cell_size
        
        # Distance from cell center
        dx = x_in_cell - 0.5
        dy = y_in_cell - 0.5
        
        if self.shape == "circle":
            dist = np.sqrt(dx**2 + dy**2)
            max_dist = 0.5
        elif self.shape == "square":
            dist = np.maximum(np.abs(dx), np.abs(dy))
            max_dist = 0.5
        elif self.shape == "diamond":
            dist = np.abs(dx) + np.abs(dy)
            max_dist = 1.0
        else:
            dist = np.sqrt(dx**2 + dy**2)
            max_dist = 0.5
        
        # Normalize distance to [0, 1]
        dist_norm = np.clip(dist / max_dist, 0.0, 1.0)
        
        # Create threshold pattern
        threshold = dist_norm ** (1.0 / self.dot_gain)
        threshold = self.min_dot_size + threshold * (self.max_dot_size - self.min_dot_size)
        
        if self.sharpness != 1.0:
            center = 0.5
            threshold = center + (threshold - center) * self.sharpness
        
        threshold = np.clip(threshold, 0.0, 1.0).astype(np.float32)
        
        return threshold, cell_assignments
    
    def _generate_halftone_screen(self, h: int, w: int) -> np.ndarray:
        """
        Generate halftone threshold screen - a fixed grid pattern like a halftone screen.
        
        Returns threshold values in [0, 1] where:
        - Center of dots = LOW threshold (0.0) - dark areas fall here
        - Edge of dots = HIGH threshold (1.0) - light areas fall here
        
        When we compare pixel brightness to this screen:
        - Dark pixels (low brightness) will be < threshold at center → use dark color
        - Light pixels (high brightness) will be >= threshold everywhere → use light color
        - Medium pixels will create the halftone dot pattern
        """
        # Convert angle to radians
        angle_rad = np.radians(self.angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Rotate coordinates
        x_rot = x_coords * cos_a - y_coords * sin_a
        y_rot = x_coords * sin_a + y_coords * cos_a
        
        # Calculate position within cell (fractional part)
        x_in_cell = (x_rot % self.cell_size) / self.cell_size  # [0, 1]
        y_in_cell = (y_rot % self.cell_size) / self.cell_size  # [0, 1]
        
        # Distance from cell center
        dx = x_in_cell - 0.5
        dy = y_in_cell - 0.5
        
        if self.shape == "circle":
            dist = np.sqrt(dx**2 + dy**2)
            max_dist = 0.5
        elif self.shape == "square":
            dist = np.maximum(np.abs(dx), np.abs(dy))
            max_dist = 0.5
        elif self.shape == "diamond":
            dist = np.abs(dx) + np.abs(dy)
            max_dist = 1.0
        else:
            dist = np.sqrt(dx**2 + dy**2)
            max_dist = 0.5
        
        # Normalize distance to [0, 1]
        dist_norm = np.clip(dist / max_dist, 0.0, 1.0)
        
        # Create threshold pattern based on distance from center
        # At center (dist=0): threshold = min_dot_size (low, for dark areas)
        # At edge (dist=1): threshold = max_dot_size (high, for light areas)
        threshold = dist_norm ** (1.0 / self.dot_gain)
        
        # Scale to the configurable range
        threshold = self.min_dot_size + threshold * (self.max_dot_size - self.min_dot_size)
        
        # Apply sharpness to make dot edges crisper
        if self.sharpness != 1.0:
            center = 0.5
            threshold = center + (threshold - center) * self.sharpness
        
        threshold = np.clip(threshold, 0.0, 1.0)
        
        return threshold.astype(np.float32)


# -------------------- Dither Utils --------------------

class DitherUtils:
    """
    Contains threshold matrices for Bayer and helper methods for gamma conversions.
    """

    BAYER2x2 = np.array([
        [0.25, 0.75],
        [1.0,  0.5]
    ], dtype=np.float32)

    BAYER4x4 = np.array([
        [0.03125, 0.53125, 0.15625, 0.65625],
        [0.78125, 0.28125, 0.90625, 0.40625],
        [0.21875, 0.71875, 0.09375, 0.59375],
        [0.96875, 0.46875, 0.84375, 0.34375]
    ], dtype=np.float32)

    BAYER8x8 = np.array([
        [0.015625, 0.515625, 0.140625, 0.640625, 0.046875, 0.546875, 0.171875, 0.671875],
        [0.765625, 0.265625, 0.890625, 0.390625, 0.796875, 0.296875, 0.921875, 0.421875],
        [0.203125, 0.703125, 0.078125, 0.578125, 0.234375, 0.734375, 0.109375, 0.609375],
        [0.953125, 0.453125, 0.828125, 0.328125, 0.984375, 0.484375, 0.84375, 0.34375],
        [0.0625,   0.5625,   0.1875,   0.6875,   0.03125,  0.53125,  0.15625,  0.65625],
        [0.8125,   0.3125,   0.9375,   0.4375,   0.78125,  0.28125,  0.90625,  0.40625],
        [0.25,      0.75,     0.125,    0.625,    0.21875,  0.71875,  0.09375,  0.59375],
        [1.0,       0.5,      0.875,    0.375,    0.96875,  0.46875,  0.84375,  0.34375]
    ], dtype=np.float32)

    BAYER16x16 = np.array([
        [0.00390625, 0.50390625, 0.12890625, 0.62890625, 0.03125, 0.53125, 0.15625, 0.65625,
         0.046875,   0.546875,   0.171875,   0.671875,   0.01171875, 0.51171875, 0.13671875, 0.63671875],
        [0.76367188, 0.26367188, 0.88867188, 0.38867188, 0.796875, 0.296875, 0.921875, 0.421875,
         0.7421875,  0.2421875,  0.8671875,  0.3671875,  0.98046875, 0.48046875, 0.90625,    0.40625],
        [0.203125,   0.703125,   0.078125,   0.578125,   0.21875,  0.71875,  0.09375,  0.59375,
         0.1484375,  0.6484375,  0.0234375,  0.5234375,  0.109375,  0.609375, 0.234375,  0.734375],
        [0.9453125,  0.4453125,  0.8203125,  0.3203125,  0.9609375, 0.4609375, 0.8359375, 0.3359375,
         0.890625,   0.390625,   0.765625,   0.265625,   0.984375,   0.484375, 0.859375,  0.359375],
        [0.0625,     0.5625,     0.1875,     0.6875,     0.03125,   0.53125,  0.15625,   0.65625,
         0.1015625,  0.6015625,  0.2265625,  0.7265625,  0.046875,   0.546875, 0.171875,  0.671875],
        [0.8125,     0.3125,     0.9375,     0.4375,     0.78125,   0.28125,  0.90625,   0.40625,
         0.8515625,  0.3515625,  0.9765625,  0.4765625,  0.796875,   0.296875, 0.921875,  0.421875],
        [0.2421875,  0.7421875,  0.1171875,  0.6171875,  0.2578125, 0.7578125, 0.1328125, 0.6328125,
         0.1484375,  0.6484375,  0.0234375,  0.5234375,  0.109375,   0.609375, 0.234375,  0.734375],
        [0.98046875, 0.48046875, 0.8671875,  0.3671875,  0.9765625, 0.4765625, 0.8515625, 0.3515625,
         0.921875,   0.421875,   0.796875,   0.296875,   0.90625,    0.40625,   0.78125,   0.28125]
    ], dtype=np.float32)

    @staticmethod
    def get_threshold_matrix(mode: DitherMode, size: str = '4x4') -> np.ndarray:
        """
        Get a threshold matrix for a given dithering mode.
        
        Args:
            mode: The dithering mode
            size: For BAYER mode, the matrix size ('2x2', '4x4', '8x8', '16x16')
        
        Returns:
            The threshold matrix as a numpy array
        """
        if mode == DitherMode.NONE:
            return np.ones((1,1), dtype=np.float32)
        elif mode == DitherMode.BAYER:
            if size == '2x2':
                return DitherUtils.BAYER2x2
            elif size == '4x4':
                return DitherUtils.BAYER4x4
            elif size == '8x8':
                return DitherUtils.BAYER8x8
            elif size == '16x16':
                return DitherUtils.BAYER16x16
            else:
                return DitherUtils.BAYER4x4  # Default fallback
        else:
            raise ValueError(f"Unsupported matrix mode: {mode}")

    @staticmethod
    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        low = (c <= 0.04045)
        out = np.empty_like(c, dtype=np.float32)
        out[low] = c[low] / 12.92
        out[~low] = ((c[~low] + 0.055) / 1.055)**2.4
        return out

    @staticmethod
    def linear_to_srgb(c: np.ndarray) -> np.ndarray:
        low = (c <= 0.0031308)
        out = np.empty_like(c, dtype=np.float32)
        out[low] = c[low] * 12.92
        out[~low] = 1.055*(c[~low]**(1.0/2.4)) - 0.055
        return out


# -------------------- ColorReducer --------------------

class ColorReducer:
    """
    Class that helps reduce an image to a smaller palette of colors.
    Includes median cut, k-means, uniform, etc.
    """

    @staticmethod
    def find_dominant_channel(colors: List[Tuple[int,int,int]]) -> int:
        rng = [0,0,0]
        for channel in range(3):
            vals = [c[channel] for c in colors]
            mn, mx = min(vals), max(vals)
            rng[channel] = mx - mn
        return rng.index(max(rng))

    @staticmethod
    def median_cut(colors: List[Tuple[int,int,int]], depth: int) -> List[Tuple[int,int,int]]:
        if depth == 0 or len(colors)==0:
            if not colors:
                return [(0,0,0)]
            avg = tuple(int(sum(c)/len(c)) for c in zip(*colors))
            return [avg]
        channel = ColorReducer.find_dominant_channel(colors)
        colors.sort(key=lambda x: x[channel])
        mid = len(colors)//2
        return (ColorReducer.median_cut(colors[:mid], depth-1) +
                ColorReducer.median_cut(colors[mid:], depth-1))

    @staticmethod
    def reduce_colors(image: Image.Image, num_colors: int) -> List[Tuple[int,int,int]]:
        """
        Reduces the given image to 'num_colors' using a naive median-cut approach.
        """
        image = image.convert('RGB')
        unique_cols = list(set(image.getdata()))
        if num_colors < 1:
            num_colors = 1
        depth = int(math.log2(num_colors)) if num_colors>1 else 0
        return ColorReducer.median_cut(unique_cols, depth)

    @staticmethod
    def generate_kmeans_palette(img: Image.Image, num_colors: int,
                                random_state=42) -> List[Tuple[int,int,int]]:
        arr = np.array(img.convert('RGB'))
        pix = arr.reshape(-1,3)
        if len(pix) > 10000:
            import random
            idx = random.sample(range(len(pix)), 10000)
            pix = pix[idx]
        km = KMeans(n_clusters=num_colors, random_state=random_state)
        km.fit(pix)
        centers = km.cluster_centers_.astype(int)
        return [tuple(c) for c in centers]

    @staticmethod
    def generate_uniform_palette(num_colors: int) -> List[Tuple[int,int,int]]:
        c = []
        cube = int(math.ceil(num_colors**(1/3)))
        for r in range(cube):
            for g in range(cube):
                for b in range(cube):
                    if len(c) >= num_colors:
                        break
                    rr = int(r*255/(cube-1)) if cube>1 else 128
                    gg = int(g*255/(cube-1)) if cube>1 else 128
                    bb = int(b*255/(cube-1)) if cube>1 else 128
                    c.append((rr,gg,bb))
        return c[:num_colors]


# -------------------- Image Ditherer --------------------

class ImageDitherer:
    """
    Orchestrates color reduction (palette building) plus dithering (using a chosen strategy).
    """
    def __init__(self,
                 num_colors: int=16,
                 dither_mode: Optional[DitherMode]=DitherMode.BAYER,
                 palette: Optional[List[Tuple[int,int,int]]]=None,
                 use_gamma: bool=False,
                 dither_params: Optional[dict]=None):
        self.num_colors = num_colors
        self.dither_mode = dither_mode
        self.palette = palette
        self.use_gamma = use_gamma
        self.dither_params = dither_params or {}
    
    @staticmethod
    def get_mode_parameters(mode: DitherMode) -> Optional[dict]:
        """
        Get parameter metadata for a specific dithering mode.
        Returns None if the mode has no configurable parameters.
        """
        if mode == DitherMode.BAYER:
            return BayerDitherStrategy.get_parameter_info()
        elif mode == DitherMode.HALFTONE:
            return HalftoneDitherStrategy.get_parameter_info()
        elif mode == DitherMode.POLKA_DOT:
            return PolkaDotDitherStrategy.get_parameter_info()
        elif mode == DitherMode.BLUE_NOISE:
            return BlueNoiseDitherStrategy.get_parameter_info()
        elif mode == DitherMode.INTERLEAVED_GRADIENT_NOISE:
            return InterleavedGradientNoiseDitherStrategy.get_parameter_info()
        elif mode == DitherMode.WAVELET:
            return WaveletDitherStrategy.get_parameter_info()
        elif mode == DitherMode.ADAPTIVE_VARIANCE:
            return AdaptiveVarianceDitherStrategy.get_parameter_info()
        elif mode == DitherMode.HYBRID:
            return HybridDitherStrategy.get_parameter_info()
        elif mode == DitherMode.ERROR_DIFFUSION:
            return ErrorDiffusionDitherStrategy.get_parameter_info()
        elif mode == DitherMode.OSTROMOUKHOV:
            return OstromoukhovDitherStrategy.get_parameter_info()
        # Add other modes here as they become configurable
        return None
    
    @staticmethod
    def mode_has_parameters(mode: DitherMode) -> bool:
        """Check if a dithering mode has configurable parameters."""
        return ImageDitherer.get_mode_parameters(mode) is not None

    def _get_dither_strategy(self, mode: DitherMode) -> BaseDitherStrategy:
        if mode == DitherMode.NONE:
            return NoDitherStrategy()
        elif mode == DitherMode.BAYER:
            # Bayer with configurable matrix size
            params = BayerDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return BayerDitherStrategy(**settings)
        elif mode == DitherMode.BLUE_NOISE:
            # Blue noise with configurable parameters
            params = BlueNoiseDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return BlueNoiseDitherStrategy(**settings)
        elif mode == DitherMode.INTERLEAVED_GRADIENT_NOISE:
            params = InterleavedGradientNoiseDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return InterleavedGradientNoiseDitherStrategy(**settings)
        elif mode == DitherMode.POLKA_DOT:
            # Polka dot with configurable parameters
            params = PolkaDotDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return PolkaDotDitherStrategy(**settings)
        elif mode == DitherMode.ERROR_DIFFUSION:
            # Unified error diffusion with configurable variant and serpentine
            params = ErrorDiffusionDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return ErrorDiffusionDitherStrategy(**settings)
        elif mode == DitherMode.RIEMERSMA:
            return RiemersmaDitherStrategy()
        elif mode == DitherMode.WAVELET:
            # Wavelet with configurable parameters
            params = WaveletDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return WaveletDitherStrategy(**settings)
        elif mode == DitherMode.ADAPTIVE_VARIANCE:
            # Adaptive variance with configurable parameters
            params = AdaptiveVarianceDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return AdaptiveVarianceDitherStrategy(**settings)
        elif mode == DitherMode.PERCEPTUAL:
            return PerceptualDitherStrategy()
        elif mode == DitherMode.HYBRID:
            # Hybrid with configurable parameters
            params = HybridDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return HybridDitherStrategy(**settings)
        elif mode == DitherMode.HALFTONE:
            # Newspaper-style halftone with configurable parameters
            # Get defaults and override with user settings
            params = HalftoneDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return HalftoneDitherStrategy(**settings)
        elif mode == DitherMode.OSTROMOUKHOV:
            # Ostromoukhov's variable error diffusion with configurable parameters
            params = OstromoukhovDitherStrategy.get_parameter_info()
            settings = {key: info['default'] for key, info in params.items()}
            settings.update(self.dither_params)
            return OstromoukhovDitherStrategy(**settings)
        else:
            raise ValueError(f"Unrecognized DitherMode: {mode}")

    def apply_dithering(self, image: Image.Image) -> Image.Image:
        arr_srgb_8 = np.array(image.convert('RGB'), dtype=np.uint8)

        # Optionally go to linear domain
        if self.use_gamma:
            arr_01 = arr_srgb_8.astype(np.float32)/255.0
            arr_lin_01 = DitherUtils.srgb_to_linear(arr_01)
            arr_lin_8 = np.clip(arr_lin_01*255.0,0,255).astype(np.uint8)
            if self.palette is None:
                temp_img_lin = Image.fromarray(arr_lin_8, 'RGB')
                self.palette = ColorReducer.reduce_colors(temp_img_lin, self.num_colors)
            arr_for_dith = arr_lin_8
        else:
            if self.palette is None:
                self.palette = ColorReducer.reduce_colors(image, self.num_colors)
            arr_for_dith = arr_srgb_8

        # Convert palette to appropriate color space
        palette_arr = np.array(self.palette, dtype=np.float32)
        if self.use_gamma:
            # Palette is in sRGB (0-255), convert to linear space for accurate matching
            palette_01 = palette_arr / 255.0
            palette_lin_01 = DitherUtils.srgb_to_linear(palette_01)
            palette_arr = np.clip(palette_lin_01 * 255.0, 0, 255).astype(np.float32)
        
        h, w, _ = arr_for_dith.shape
        flat_pixels = arr_for_dith.reshape((-1, 3)).astype(np.float32)

        if not self.dither_mode:
            self.dither_mode = DitherMode.NONE

        strategy = self._get_dither_strategy(self.dither_mode)
        dithered_flat = strategy.dither(flat_pixels, palette_arr, (h,w))
        out_arr_lin_8 = dithered_flat.reshape((h,w,3)).astype(np.uint8)

        if self.use_gamma:
            out_lin_01 = out_arr_lin_8.astype(np.float32)/255.0
            out_srgb_01 = DitherUtils.linear_to_srgb(np.clip(out_lin_01,0,1))
            out_srgb_8 = np.clip(out_srgb_01*255.0, 0,255).astype(np.uint8)
            return Image.fromarray(out_srgb_8, 'RGB')
        else:
            return Image.fromarray(out_arr_lin_8, 'RGB')
