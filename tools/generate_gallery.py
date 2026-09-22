"""Regenerate the bundled algorithm examples; not run at application startup."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from PIL import Image
from dithering_lib import DitherMode, ImageDitherer


def main():
    destination = Path(__file__).resolve().parents[1] / 'ui' / 'modern' / 'assets' / 'gallery'
    destination.mkdir(parents=True, exist_ok=True)
    source = Image.new('RGB', (48, 32))
    source.putdata([(int(x * 255 / 47),) * 3 for y in range(32) for x in range(48)])
    manifest = {}
    for mode in DitherMode:
        parameters = {'size': 32} if mode == DitherMode.BLUE_NOISE else {}
        result = ImageDitherer(num_colors=2, dither_mode=mode,
            palette=[(0, 0, 0), (255, 255, 255)], dither_params=parameters).apply_dithering(source)
        result.resize((192, 128), Image.Resampling.NEAREST).save(destination / f'{mode.value}.png')
        manifest[mode.value] = {'parameters': parameters, 'file': f'{mode.value}.png'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(manifest)} gallery examples')


if __name__ == '__main__':
    main()
