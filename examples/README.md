# Dither Pie CLI - Example Configurations

This folder contains example configuration files for various use cases.

## Basic Examples

### `image_basic.json`
Simple image dithering without pixelization. Good starting point.
```bash
python ../dither_pie.py image_basic.json
```

### `image_pixelized.json`
Classic retro pixel art effect with dithering and upscaling.
```bash
python ../dither_pie.py image_pixelized.json
```

### `image_neural.json`
AI-powered neural pixelization for organic subjects. Slower but higher quality.
```bash
python ../dither_pie.py image_neural.json
```

### `image_custom_palette.json`
Using a predefined palette from `palette.json` (Game Boy DMG in this example).
```bash
python ../dither_pie.py image_custom_palette.json
```

## Video Processing

### `video_basic.json`
Process a video with pixelization and dithering. Audio is preserved.
```bash
python ../dither_pie.py video_basic.json
```

## Batch Processing

### `batch_folder.json`
Process all images in a folder at once. Great for bulk processing.
```bash
python ../dither_pie.py batch_folder.json
```

## Input Override Mode

### `settings_override.json`
Use this config with any file/folder as a second argument. The output filename is automatically generated based on settings.

```bash
# Process a specific file (output saved in same directory)
python ../dither_pie.py settings_override.json ../test_300.png
# Output: ../test_300_bayer_16c.png

# Process a folder (creates folder_processed directory)
python ../dither_pie.py settings_override.json ../misc/
# Output: ../misc_processed/

# Works with videos too
python ../dither_pie.py settings_override.json ../video.mp4
# Output: ../video_bayer_16c.mp4
```

**Output Naming Format:**
- `originalname_dithermode_numcolors.ext`
- Pixelization: adds `pix64` (if enabled)
- Gamma: adds `gamma` (if enabled)
- Palette names: `gameboy` instead of color count
- Example: `photo_pix64_error_diffusion_gameboy_gamma.png`

---

## Palette Sources

You can use different palette generation methods in the `"palette"` section:

- **`"median_cut"`** - Classic color quantization (fast)
- **`"kmeans"`** - Machine learning clustering (better quality)
- **`"uniform"`** - Evenly distributed colors
- **`"file:path/to/image.png"`** - Extract palette from another image
- **`"gameboy_dmg"`** - Use predefined palette from palette.json (replace with any palette name)
- **`"custom:palette_name"`** - Explicit custom palette reference

**Note:** For custom/predefined palettes, `num_colors` is ignored (uses palette's actual color count).

---

## Dithering Modes

Available dithering algorithms (use in `"dithering.mode"`):

- `none` - No dithering (quantize only)
- `bayer` - Ordered dithering (classic, fast)
- `error_diffusion` - Floyd-Steinberg and variants (high quality)
- `blue_noise` - High-quality spatial distribution
- `halftone` - Newspaper-style printing effect
- `riemersma` - Hilbert curve-based
- `ostromoukhov` - Adaptive error diffusion
- `polka_dot` - Retro circular patterns
- `wavelet` - Multi-scale frequency decomposition
- `adaptive_variance` - Context-aware
- `perceptual` - Luminance-preserving
- `hybrid` - Separates luminance/color channels

---

## Tips

1. **Paths**: All paths are relative to the config file location
2. **Testing**: Try processing a single image before batch processing
3. **Gamma**: Enable `"use_gamma": true` for more accurate color perception
4. **Neural**: Neural pixelization is slower but produces better results for photos
5. **Final Resize**: Use integer multipliers to preserve pixel-perfect scaling
6. **Verbose Output**: Run with `-v` flag for detailed logging: `python dither_pie.py -v config.json`

---

## Quick Start

1. Copy an example config
2. Modify input/output paths
3. Adjust parameters as needed
4. Run: `python ../dither_pie.py your_config.json`

For more information, see `../README_NEW.md`

