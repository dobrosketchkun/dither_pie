"""Clip-level pixel-grid/palette analysis and deterministic frame snapping."""
from dataclasses import dataclass, field, replace
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from pixel_snapper import SnapSettings, SnapResult, _edge_profiles, _estimate_spacing, _quantize


def is_animated_gif(path):
    if Path(path).suffix.lower() != '.gif':
        return False
    try:
        with Image.open(path) as image:
            return bool(getattr(image, 'is_animated', False))
    except OSError:
        # Leave ordinary image loading to report unreadable/corrupt files.
        return False


def gif_info(path):
    with Image.open(path) as image:
        durations = []
        has_alpha = False
        loop = image.info.get('loop')
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(max(10, int(image.info.get('duration', 100))))
            has_alpha |= image.convert('RGBA').getchannel('A').getextrema()[0] < 255
        total = sum(durations)
        rate = Fraction(image.n_frames * 1000, total)
        return dict(width=image.width, height=image.height, fps=float(rate), fps_fraction=str(rate),
                    frame_count=image.n_frames, duration=total / 1000, durations=tuple(durations), loop=loop,
                    has_alpha=has_alpha)


def read_clip_frame(path, info, index=0):
    if Path(path).suffix.lower() == '.gif':
        with Image.open(path) as image:
            image.seek(index)
            return image.convert('RGBA')
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(index / info['fps']),
        '-i', str(path), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
        capture_output=True, check=True)
    with Image.open(BytesIO(result.stdout)) as image:
        return image.convert('RGB')


@dataclass
class VideoSnapPlan:
    source_size: tuple
    columns: tuple
    rows: tuple
    pixel_size: float
    palette: tuple
    settings: SnapSettings
    clean: bool = False
    has_alpha: bool = False
    lookup_palette: tuple | None = None
    preserve_colors: bool = False
    _maps: tuple = field(init=False, repr=False)
    _tree: object = field(init=False, repr=False)

    def __post_init__(self):
        if not 1 <= len(self.palette) <= 256:
            raise ValueError('A video palette must contain 1–256 colors.')
        cols, rows = np.asarray(self.columns), np.asarray(self.rows)
        maps = []
        fractions = (.5,) if self.clean else (.3, .5, .7)
        for fy in fractions:
            for fx in fractions:
                xs = np.minimum(cols[:-1] + (np.diff(cols) * fx).astype(int), cols[-1] - 1)
                ys = np.minimum(rows[:-1] + (np.diff(rows) * fy).astype(int), rows[-1] - 1)
                maps.append((xs, ys))
        self._maps = tuple(maps)
        self._tree = cKDTree(self.lookup_palette or self.palette)

    @property
    def output_size(self):
        return len(self.columns) - 1, len(self.rows) - 1

    def sample(self, image):
        if image.size != self.source_size:
            raise ValueError('Video frame dimensions differ from the analyzed clip.')
        rgba = np.asarray(image.convert('RGBA'))
        samples = [rgba[ys[:, None], xs[None, :]] for xs, ys in self._maps]
        return samples[0].copy() if self.clean else np.median(np.stack(samples, axis=2), axis=2).astype(np.uint8)

    def render(self, image):
        rgba = self.sample(image)
        visible = rgba[..., 3] > 0
        if not self.preserve_colors and np.any(visible):
            _, indices = self._tree.query(rgba[..., :3][visible])
            rgba[..., :3][visible] = np.asarray(self.palette, dtype=np.uint8)[indices]
        rgba[~visible] = 0
        result = Image.fromarray(rgba, 'RGBA')
        return result.convert('RGB') if np.all(rgba[..., 3] == 255) else result

    def preview(self, image):
        return SnapResult(self.render(image), self.pixel_size, self.columns, self.rows,
            self.settings.pixel_size is None, '', self.palette,
            ('Original colors · palette edits enable reduction' if self.preserve_colors
             else f'Locked for clip · {len(self.palette)} shared colors'))

    def with_palette(self, palette, custom=False):
        palette = tuple(tuple(color) for color in palette)
        if not custom and len(palette) != len(self.palette):
            raise ValueError('Replacement palette size changed.')
        return replace(self, palette=palette, preserve_colors=False,
                       lookup_palette=None if custom else self.lookup_palette or self.palette)


def _locked_grid(frames, settings):
    profiles = [_edge_profiles(np.asarray(image.convert('RGBA'))) for image in frames]
    averaged = [np.mean([p[axis] for p in profiles], axis=0) for axis in (0, 1)]
    if settings.pixel_size is not None:
        step = float(settings.pixel_size)
    else:
        guesses = [v for profile in averaged if (v := _estimate_spacing(profile)) is not None]
        if not guesses:
            raise ValueError('No reliable grid found in this clip. Set a manual pixel size.')
        guess, estimates = float(np.median(guesses)), []
        for profile in averaged:
            if profile.sum() == 0:
                continue
            n = len(profile)
            frequencies = np.fft.rfftfreq(n * 64)
            magnitude = np.abs(np.fft.rfft(profile - profile.mean(), n=n * 64))
            magnitude[(frequencies <= 1 / (guess * 1.35)) | (frequencies >= 1 / (guess * .75))] = 0
            index = int(magnitude.argmax())
            if magnitude[index] / profile.sum() >= .1:
                estimates.append(1 / frequencies[index])
        if not estimates:
            raise ValueError('Grid periodicity is uncertain. Set a manual pixel size.')
        step = float(np.mean(estimates))
        # Short frames have broader spectral peaks. Prefer the simplest ratio
        # within the uncertainty of the measured period rather than fitting a
        # high-denominator fraction to that finite-frame bias.
        tolerance = max(step * .001, step * step / (max(len(p) for p in averaged) * 8))
        for denominator in range(1, 33):
            rational = round(step * denominator) / denominator
            if abs(rational - step) <= tolerance:
                step = rational
                break
    if step > min(frames[0].size):
        raise ValueError('Pixel size cannot exceed the shorter frame edge.')
    cuts = []
    for profile in averaged:
        phase = ((-np.angle(np.sum(profile * np.exp(-2j * np.pi * np.arange(len(profile)) / step)))
                  * step / (2 * np.pi)) % step) if profile.sum() else 0
        if min(phase, step - phase) < .1:
            phase = 0
        boundaries = np.rint(phase + np.arange(int(len(profile) / step) + 2) * step).astype(int)
        cuts.append(tuple(sorted({0, len(profile), *(int(v) for v in boundaries if 0 < v < len(profile))})))
    return cuts[0], cuts[1], step


def analyze_video_snap(path, settings=None, progress=None):
    """Analyze five frames once; GIF palettes additionally include all frames."""
    from video_processor import VideoProcessor
    settings = settings or SnapSettings()
    progress = progress or (lambda fraction, text: None)
    info = VideoProcessor().get_video_info(path)
    count = info['frame_count']
    if not count:
        # Duration-based samples still work when a container omits frame count.
        count = max(1, round(info['duration'] * info['fps']))
    indices = sorted({round(i * (count - 1) / 4) for i in range(5)})
    frames, gif_colors, has_alpha = [], set(), False
    clean = Path(path).suffix.lower() == '.gif'
    if clean:
        with Image.open(path) as animation:
            for index in range(animation.n_frames):
                animation.seek(index)
                rgba = animation.convert('RGBA')
                has_alpha |= rgba.getchannel('A').getextrema()[0] < 255
                colors = rgba.getcolors(257)
                if gif_colors is not None:
                    if colors is None:
                        gif_colors = None
                    else:
                        gif_colors.update(color[:3] for _, color in colors if color[3])
                        if len(gif_colors) > 256:
                            gif_colors = None
                if index in indices:
                    frames.append(rgba)
                progress(.1 + .3 * (index + 1) / animation.n_frames, 'Analyzing animation palette…')
    else:
        for i, index in enumerate(indices):
            progress(.1 + .3 * i / len(indices), 'Analyzing clip grid and palette…')
            frames.append(read_clip_frame(path, info, index))
    progress(.45, 'Locking pixel grid…')
    cols, rows, step = _locked_grid(frames, settings)
    provisional = VideoSnapPlan(frames[0].size, cols, rows, step, ((0, 0, 0),), settings, clean, has_alpha)
    # Auto is a preservation mode, not an implicit 256-color budget. Compressed
    # video often contains gradients with thousands of legitimate shades. Keep
    # sampled RGB values in that case; the palette remains an editable suggestion.
    exact_colors = gif_colors if clean else set()
    if not clean and settings.num_colors is None:
        for frame in frames:
            colors = frame.convert('RGB').getcolors(256)
            if colors is None:
                exact_colors = None
                break
            exact_colors.update(color for _, color in colors)
            if len(exact_colors) > 256:
                exact_colors = None
                break
    preserve_colors = settings.num_colors is None and not clean and exact_colors is None
    if settings.num_colors is None and exact_colors:
        palette = tuple(sorted(exact_colors))
    else:
        samples = [provisional.sample(image).reshape(-1, 4) for image in frames]
        pixels = np.concatenate(samples)
        # Bound palette analysis independently of the number/resolution of frames.
        if len(pixels) > 32768:
            pixels = pixels[np.random.default_rng(42).choice(len(pixels), 32768, replace=False)]
        palette_settings = replace(settings, num_colors=256) if settings.num_colors is None else settings
        _, palette, _, _ = _quantize(pixels.reshape(1, -1, 4), palette_settings)
        if not palette:
            palette = ((0, 0, 0),)
    progress(1, 'Clip grid and palette ready')
    return replace(provisional, palette=palette, preserve_colors=preserve_colors)
