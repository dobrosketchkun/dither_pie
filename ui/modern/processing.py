"""Snapshot-based operations for the production interface. No Tk calls here."""
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from dithering_lib import ColorReducer, DitherMode, ImageDitherer
from utils import split_alpha, has_actual_transparency


GENERATED = ('Median Cut', 'K-means', 'Uniform')


@dataclass(frozen=True)
class DitherRequest:
    mode: str
    count: int
    gamma: bool
    palette_name: str
    palette: tuple | None
    parameters: dict

    def render(self, source):
        colors = self.palette
        if colors is None:
            rgb, alpha = split_alpha(source)
            mask = None if alpha is None else alpha > 0
            if self.palette_name == 'Uniform':
                colors = ColorReducer.generate_uniform_palette(self.count)
            elif self.palette_name == 'K-means':
                colors = ColorReducer.generate_kmeans_palette(rgb, self.count, random_state=42, mask=mask)
            else:
                colors = ColorReducer.reduce_colors(rgb, self.count, mask=mask)
        colors = [tuple(int(v) for v in c) for c in colors]
        result = self.ditherer(colors).apply_dithering(source)
        return result, colors

    def ditherer(self, palette):
        return ImageDitherer(num_colors=self.count, dither_mode=DitherMode(self.mode),
                             palette=palette, use_gamma=self.gamma,
                             dither_params=dict(self.parameters))


def read_image(path):
    with Image.open(path) as image:
        return image.convert('RGBA' if has_actual_transparency(image) else 'RGB')


def read_video_frame(path, info, index=0):
    rate = float(info['fps_fraction'])
    command = ['ffmpeg', '-v', 'error', '-ss', str(index / rate), '-i', str(path),
               '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-']
    result = subprocess.run(command, capture_output=True, check=True)
    from io import BytesIO
    with Image.open(BytesIO(result.stdout)) as image:
        return image.convert('RGB')
