"""Bounded-memory animated GIF export with a global palette and exact timing."""
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, GifImagePlugin
from scipy.spatial import cKDTree


def process_gif(source, destination, ditherer, method, max_size, multiplier, snapper_plan, progress):
    from pixelization import pixelize_regular
    palette_plan = snapper_plan
    if palette_plan is None and ditherer is None:
        from video_snapper import analyze_video_snap
        from pixel_snapper import SnapSettings
        palette_plan = analyze_video_snap(source, SnapSettings(pixel_size=1),
            lambda f, text: progress(.05 * f, text))
    if palette_plan is not None:
        transparent = palette_plan.has_alpha
    else:
        from video_snapper import gif_info
        transparent = gif_info(source)['has_alpha']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dither_', dir=destination.parent) as directory:
        staged = Path(directory) / 'animation.gif'
        with Image.open(source) as animation, staged.open('wb') as output:
            loop = animation.info.get('loop')
            palette = tree = None
            for index in range(animation.n_frames):
                animation.seek(index)
                duration = max(10, int(animation.info.get('duration', 100)))
                image = animation.convert('RGBA')
                if snapper_plan is not None:
                    image = snapper_plan.render(image)
                elif method == 'regular':
                    image = pixelize_regular(image, max_size)
                elif method == 'neural':
                    raise ValueError('Neural pixelization cannot preserve GIF transparency; choose Regular or Snapper.')
                if ditherer is not None:
                    image = ditherer.apply_dithering(image)
                if multiplier:
                    image = image.resize((image.width * multiplier, image.height * multiplier), Image.Resampling.NEAREST)
                rgba = np.asarray(image.convert('RGBA'))
                if palette is None:
                    colors = (ditherer.palette if ditherer is not None else
                              palette_plan.palette if palette_plan is not None else None)
                    if not colors:
                        raise ValueError('GIF export requires Snapper or a selected shared dithering palette.')
                    # Reserve a stable transparency index if any source frame
                    # needs it, even when the first frame is fully opaque.
                    palette = np.asarray(colors[:255 if transparent else 256], dtype=np.uint8)
                    tree = cKDTree(palette)
                    palette_bytes = np.pad(palette.reshape(-1), (0, 768 - palette.size)).tolist()
                visible = rgba[..., 3] > 128
                indices = np.full(rgba.shape[:2], 255 if transparent else 0, dtype=np.uint8)
                if visible.any():
                    _, chosen = tree.query(rgba[..., :3][visible])
                    indices[visible] = chosen
                frame = Image.fromarray(indices, 'P')
                frame.putpalette(palette_bytes)
                if index == 0:
                    info = {'optimize': False, 'duration': duration}
                    if transparent:
                        info.update(transparency=255, background=255)
                    if loop is not None:
                        info['loop'] = loop
                    header, _ = GifImagePlugin.getheader(frame, info=info)
                    for block in header:
                        output.write(block)
                options = {'transparency': 255} if transparent else {}
                for block in GifImagePlugin.getdata(frame, duration=duration, disposal=2, **options):
                    output.write(block)
                progress(.1 + .85 * (index + 1) / animation.n_frames, f'Processed {index + 1} GIF frames')
            output.write(b';')
        os.replace(staged, destination)
    progress(1, 'Animation processing complete!')
    return True
