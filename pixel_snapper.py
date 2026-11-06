"""Recover an uneven source pixel grid and reduce each cell to one pixel.

Inspired by Hugo Duprez's Sprite Fusion Pixel Snapper: edge projections,
spacing estimation, adaptive boundary walking, and dominant-cell sampling.
This implementation uses RGB/alpha edges and deterministic palette reduction.
No GUI dependencies; source images are never modified.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


QUANTIZERS = ('median_cut', 'kmeans', 'octree')


@dataclass(frozen=True)
class SnapSettings:
    num_colors: int | None = None
    quantizer: str = 'median_cut'
    pixel_size: float | None = None

    def __post_init__(self):
        if self.num_colors is not None and (isinstance(self.num_colors, bool) or not isinstance(self.num_colors, int) or not 1 <= self.num_colors <= 256):
            raise ValueError('Color count must be an integer from 1 to 256.')
        if self.quantizer not in QUANTIZERS:
            raise ValueError(f'Unknown color reduction algorithm: {self.quantizer}')
        if self.pixel_size is not None:
            if (isinstance(self.pixel_size, bool) or not isinstance(self.pixel_size, (int, float))
                    or not math.isfinite(self.pixel_size) or self.pixel_size < 1):
                raise ValueError('Pixel size must be a finite number of at least 1, or null for automatic detection.')


@dataclass(frozen=True)
class SnapResult:
    image: Image.Image
    pixel_size: float
    columns: tuple
    rows: tuple
    automatic: bool
    warning: str = ''
    palette: tuple = ()
    color_note: str = ''


def _estimate_color_count(sample):
    """Find a compact count with low RGB error on a bounded analysis sample.

    This is a fidelity heuristic for high-color/noisy sources, not a claim to
    recover the original palette size from a compressed image.
    """
    strip = Image.fromarray(sample.reshape(1, -1, 3), 'RGB')
    for count in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        indexed = strip.quantize(colors=count, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        error = np.asarray(strip, dtype=np.float32) - np.asarray(indexed.convert('RGB'), dtype=np.float32)
        distances = np.sqrt(np.sum(error * error, axis=2))
        if np.sqrt(np.mean(error * error)) <= 3 and np.percentile(distances, 95) <= 12:
            return count
    return 256


def _build_palette(sample, count, quantizer):
    if quantizer == 'kmeans':
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
        count = min(count, len(np.unique(sample, axis=0)))
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=count, random_state=42, n_init=3).fit(sample.astype(np.float32))
        return np.rint(model.cluster_centers_).clip(0, 255).astype(np.uint8)
    strip = Image.fromarray(sample.reshape(1, -1, 3), 'RGB')
    method = Image.Quantize.MEDIANCUT if quantizer == 'median_cut' else Image.Quantize.FASTOCTREE
    indexed = strip.quantize(colors=count, method=method, dither=Image.Dither.NONE)
    used = sorted(indexed.getcolors(), key=lambda item: item[1])
    raw = np.asarray(indexed.getpalette(), dtype=np.uint8).reshape(-1, 3)
    return raw[[index for _, index in used]]


def _quantize(rgba, settings):
    visible = rgba[..., 3] > 0
    result = rgba.copy()
    result[~visible] = 0
    pixels = rgba[..., :3][visible]
    if not len(pixels):
        return result, (), 'No visible colors', 0
    # Only visible colors participate. Padding repeats visible samples, never
    # hidden RGB, and the bounded sample keeps large images practical.
    unique = np.unique(pixels, axis=0)
    automatic = settings.num_colors is None
    count = min(settings.num_colors or 256, len(unique))
    if count == len(unique):
        note = f'Preserved all {count} source colors' if automatic else ''
        return result, tuple(map(tuple, unique.tolist())), note, len(unique)
    rng = np.random.default_rng(42)
    sample = pixels if len(pixels) <= 32768 else pixels[rng.choice(len(pixels), 32768, replace=False)]
    if automatic:
        count = _estimate_color_count(sample)
    while True:
        palette = _build_palette(sample, count, settings.quantizer)
        tree = cKDTree(palette.astype(np.float32))
        if not automatic or count == 256:
            break
        # Check the selected algorithm as well: Octree and K-means do not
        # produce the same error as the Median Cut estimate at a given count.
        distances, _ = tree.query(sample)
        if np.sqrt(np.mean(distances * distances) / 3) <= 3 and np.percentile(distances, 95) <= 12:
            break
        count = next(c for c in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if c > count)
    # Bound the query workspace independently of image dimensions.
    mapped = np.empty_like(pixels)
    for start in range(0, len(pixels), 262144):
        chunk = pixels[start:start + 262144]
        _, indices = tree.query(chunk)
        mapped[start:start + len(chunk)] = palette[indices]
    result[..., :3][visible] = mapped
    note = f'Estimated {len(palette)} colors (source has more than 256)' if automatic else ''
    return result, tuple(map(tuple, palette.tolist())), note, len(unique)


def _edge_profiles(rgba):
    # Premultiplication removes hidden RGB and keeps color edges at comparable
    # strength across transparency; alpha itself contributes silhouette edges.
    data = rgba.astype(np.float32)
    data[..., :3] *= data[..., 3:4] / 255.0
    dx = np.abs(np.diff(data, axis=1)).sum(axis=2).sum(axis=0)
    dy = np.abs(np.diff(data, axis=0)).sum(axis=2).sum(axis=1)
    return np.r_[0.0, dx], np.r_[0.0, dy]


def _estimate_spacing(profile):
    if len(profile) < 3 or profile.max() == 0:
        return None
    # Padding includes boundaries near the image edges. find_peaks handles
    # plateaus, which occur naturally on antialiased and clean enlarged art.
    peaks, _ = find_peaks(np.r_[0, profile, 0], height=profile.max() * .2, distance=2)
    peaks = peaks - 1
    if len(peaks) < 3:
        return None
    gaps = np.diff(peaks).astype(float)
    step = float(np.median(gaps))
    # Multiples are expected when adjacent source pixels have the same color.
    residual = np.abs(gaps / step - np.rint(gaps / step))
    if np.mean(residual <= .22) < .6:
        return None
    return max(1.0, step)


def _walk(profile, step):
    limit = len(profile)
    if step <= 1:
        return tuple(range(limit + 1))
    cuts = [0]
    threshold = float(profile.mean()) * .5
    while cuts[-1] + step < limit - .35 * step:
        target = cuts[-1] + step
        radius = max(1, step * .2)
        lo = max(cuts[-1] + 1, math.ceil(target - radius))
        hi = min(limit - 1, math.floor(target + radius))
        candidates = np.arange(lo, hi + 1)
        if len(candidates) and profile[candidates].max() > threshold:
            # Prefer the nearest boundary when equally strong edges compete.
            scores = profile[candidates] - np.abs(candidates - target) * 1e-6
            cut = int(candidates[np.argmax(scores)])
        else:
            cut = min(limit - 1, max(cuts[-1] + 1, round(target)))
        cuts.append(cut)
    cuts.append(limit)
    return tuple(cuts)


def _sample_grid(rgba, cols, rows):
    if len(cols) == rgba.shape[1] + 1 and len(rows) == rgba.shape[0] + 1:
        return rgba.copy()
    output = np.zeros((len(rows) - 1, len(cols) - 1, 4), dtype=np.uint8)
    for y, (top, bottom) in enumerate(zip(rows, rows[1:])):
        for x, (left, right) in enumerate(zip(cols, cols[1:])):
            cell = rgba[top:bottom, left:right]
            colors, counts = np.unique(cell.reshape(-1, 4), axis=0, return_counts=True)
            winners = colors[counts == counts.max()]
            center = cell[(bottom - top) // 2, (right - left) // 2]
            output[y, x] = center if np.all(winners == center, axis=1).any() else winners[0]
    return output


def _recover_block_colors(rgba, cols, rows):
    """Recover compressed enlarged pixel art only on a strong regular grid.

    Cell interiors avoid JPEG edge ringing. A radius-limited palette merges
    small color variations while retaining rare colors regardless of area.
    Uncertain grids and palettes exceeding 256 colors use the general path.
    """
    widths, heights = np.diff(cols), np.diff(rows)
    if (len(widths) < 2 or len(heights) < 2 or min(widths.min(), heights.min()) < 4 or
            np.any(widths != widths[0]) or np.any(heights != heights[0])):
        return None
    px, py = _edge_profiles(rgba)
    for profile, cuts in ((px, cols), (py, rows)):
        total = float(profile.sum())
        if total == 0 or profile[np.asarray(cuts[1:-1])].sum() / total < .65:
            return None
    sx, sy = int(widths[0]), int(heights[0])
    cells = rgba.reshape(len(heights), sy, len(widths), sx, 4).transpose(0, 2, 1, 3, 4)
    core = cells[:, :, sy // 4:sy - sy // 4, sx // 4:sx - sx // 4]
    output = np.rint(np.median(core, axis=(2, 3))).astype(np.uint8)
    visible = output[..., 3] > 0
    output[~visible] = 0
    colors, counts = np.unique(output[..., :3][visible], axis=0, return_counts=True)
    if not len(colors):
        return None
    palette = []
    # No frequency cutoff: a single-pixel eye highlight has the same maximum
    # allowed color error as a large background. Centers are observed colors.
    for index in np.argsort(-counts, kind='stable'):
        color = colors[index].astype(np.float32)
        if not palette or np.min(np.sum((np.asarray(palette) - color) ** 2, axis=1)) > 6 ** 2:
            palette.append(color)
            if len(palette) > 256:
                return None
    _, indices = cKDTree(palette).query(output[..., :3][visible])
    output[..., :3][visible] = np.asarray(palette, dtype=np.uint8)[indices]
    colors = tuple(tuple(int(v) for v in color) for color in palette)
    return output, colors, f'Recovered approximately {len(colors)} block colors (compression cleanup)'


def apply_snap_palette(source, base, palette, *, custom=False):
    """Recolor a cached grid without regenerating its palette or boundaries.

    Replacements retain cluster membership. Adding/deleting colors switches to
    nearest-color quantization of the original source using the edited palette.
    """
    if (not 1 <= len(palette) <= 256 or any(len(color) != 3 or
            any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in color)
            for color in palette)):
        raise ValueError('Palette must contain 1–256 RGB colors.')
    if custom:
        rgba = np.array(source.convert('RGBA'))
        visible = rgba[..., 3] > 0
        pixels = rgba[..., :3][visible]
        mapped = np.empty_like(pixels)
        tree = cKDTree(palette)
        for start in range(0, len(pixels), 262144):
            chunk = pixels[start:start + 262144]
            _, indices = tree.query(chunk)
            mapped[start:start + len(chunk)] = np.asarray(palette, dtype=np.uint8)[indices]
        rgba[..., :3][visible] = mapped
        rgba[~visible] = 0
        output = _sample_grid(rgba, base.columns, base.rows)
    else:
        if len(palette) != len(base.palette):
            raise ValueError('Replacement palette must match the generated palette size.')
        output = np.array(base.image.convert('RGBA'))
        visible = output[..., 3] > 0
        if base.palette:
            _, indices = cKDTree(base.palette).query(output[..., :3][visible])
            output[..., :3][visible] = np.asarray(palette, dtype=np.uint8)[indices]
    image = Image.fromarray(output, 'RGBA')
    if np.all(output[..., 3] == 255):
        image = image.convert('RGB')
    return SnapResult(image, base.pixel_size, base.columns, base.rows, base.automatic,
                      base.warning, tuple(map(tuple, palette)), f'Edited palette: {len(palette)} colors')


def snap_pixels(image: Image.Image, settings: SnapSettings | None = None, *, palette=None) -> SnapResult:
    """Return a logical-resolution image, retaining naturally odd dimensions.

    Automatic detection requires repeated edges. Ambiguous images require a
    manual cell size rather than silently falling back to a destructive guess.
    Cell colors are modes of quantized RGBA values, with center-biased ties.
    """
    settings = settings or SnapSettings()
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError('Image dimensions must be positive.')
    if settings.pixel_size is not None and settings.pixel_size > min(width, height):
        raise ValueError('Pixel size cannot exceed the shorter image edge.')
    source_rgba = np.asarray(image.convert('RGBA'))
    rgba, generated_palette, color_note, source_colors = _quantize(source_rgba, settings)
    px, py = _edge_profiles(rgba)
    automatic = settings.pixel_size is None
    warning = ''
    if automatic:
        sx, sy = _estimate_spacing(px), _estimate_spacing(py)
        if sx is None and sy is None:
            raise ValueError('No reliable repeated grid was found. Set a manual pixel size and preview again.')
        if sx is None or sy is None:
            step = sx if sx is not None else sy
            warning = 'Grid spacing could only be detected on one axis; check the preview.'
        elif max(sx, sy) / min(sx, sy) > 1.5:
            step = min(sx, sy)
            warning = 'The axes suggest different cell sizes; check the preview or set a manual size.'
        else:
            step = (sx + sy) / 2
    else:
        step = float(settings.pixel_size)
    cols, rows = _walk(px, step), _walk(py, step)
    recovered = (_recover_block_colors(source_rgba, cols, rows)
                 if settings.num_colors is None and source_colors > 256 else None)
    if recovered is not None:
        output, generated_palette, color_note = recovered
    else:
        output = _sample_grid(rgba, cols, rows)
    if palette is not None:
        if (len(palette) != len(generated_palette) or
                any(len(color) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255
                                           for v in color) for color in palette)):
            raise ValueError('An edited palette must contain one RGB color per generated palette entry.')
    if palette is not None and generated_palette:
        # Editing a swatch recolors its existing cluster. Do not reassign pixels
        # to a different cluster or move the detected grid as colors are edited.
        visible = output[..., 3] > 0
        _, indices = cKDTree(generated_palette).query(output[..., :3][visible])
        output[..., :3][visible] = np.asarray(palette, dtype=np.uint8)[indices]
    result = Image.fromarray(output, 'RGBA')
    if np.all(output[..., 3] == 255):
        result = result.convert('RGB')
    return SnapResult(result, float(step), cols, rows, automatic, warning,
                      generated_palette if palette is None else tuple(map(tuple, palette)),
                      color_note if palette is None else f'Edited palette: {len(palette)} colors')
