# Dither Pie CLI - Usage Guide

Complete guide for using Dither Pie from the command line.

---

## Quick Start

```bash
# Show help
python dither_pie.py --help

# Generate example config
python dither_pie.py --example-config > my_config.json

# Process with config file
python dither_pie.py my_config.json

# Process specific file with config settings (input override)
python dither_pie.py my_config.json input_image.png

# Process folder with config settings (input override)
python dither_pie.py my_config.json my_images/

# Verbose output
python dither_pie.py -v my_config.json

# Log to file
python dither_pie.py --log-file processing.log my_config.json
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message |
| `--example-config` | Generate example configuration |
| `--verbose`, `-v` | Enable verbose (DEBUG) logging |
| `--quiet`, `-q` | Quiet mode (errors only) |
| `--log-file FILE` | Write log to file |

---

## Input Override Feature

You can use a config file's settings with **different input files** by providing a second argument:

```bash
python dither_pie.py config.json <input_file_or_folder>
```

**What happens:**
- Config settings (dithering, palette, etc.) are used from `config.json`
- Input/output paths from config are **ignored**
- Input is taken from the second argument
- Output is **auto-generated** with descriptive filename

**Examples:**

```bash
# Apply settings to specific image (auto-generates output name)
python dither_pie.py settings.json photo.jpg
# Output: photo_bayer_16c.jpg (in same directory)

# Apply settings to folder (creates processed folder)
python dither_pie.py settings.json images/
# Output: images_processed/ folder

# With different file types
python dither_pie.py settings.json video.mp4
# Output: video_pixelized_bayer_16c.mp4
```

**Auto-generated filenames include:**
- Base filename (truncated to 30 chars)
- Pixelization method and size (if enabled): `pix64`, `pix150`
- Dither mode: `bayer`, `error_diffusion`, etc.
- Palette info: `16c`, `km32c`, `gameboy_dmg`
- Gamma: `gamma` (if enabled)

**Example output names:**
- `sunset_pix128_bayer_16c.png`
- `portrait_neural_pix64_error_diffusion_32c_gamma.png`
- `video_bayer_gameboy_dmg.mp4`

---

## Configuration File Format

All processing parameters are specified in a JSON configuration file.

### Minimal Configuration

```json
{
    "input": "path/to/input.png",
    "output": "path/to/output.png"
}
```

This will:
- Auto-detect mode (image/video/folder)
- Skip pixelization
- Apply Bayer dithering
- Use median-cut palette with 16 colors

**Note:** When using input override (second argument), the `input` and `output` fields are ignored and can be set to dummy values like `"input.png"` and `"output.png"`.

### Full Configuration

```json
{
    "input": "path/to/input.png",
    "output": "path/to/output.png",
    "mode": "image",
    "pixelization": {
        "enabled": true,
        "method": "regular",
        "max_size": 128
    },
    "dithering": {
        "enabled": true,
        "mode": "bayer",
        "parameters": {}
    },
    "palette": {
        "source": "median_cut",
        "num_colors": 16,
        "use_gamma": false
    },
    "final_resize": {
        "enabled": false,
        "multiplier": 2
    }
}
```

---

## Configuration Fields

### Required Fields

- **`input`**: Path to input file or directory (ignored if using input override)
- **`output`**: Path to output file or directory (ignored if using input override)

### Optional Fields

#### `mode` (auto-detected if not specified)
- `"image"` - Process single image
- `"video"` - Process single video
- `"folder"` - Batch process folder

#### `pixelization`
- **`enabled`**: `true` or `false`
- **`method`**: `"none"`, `"regular"`, or `"neural"`
- **`max_size`**: Maximum dimension (integer)

#### `dithering`
- **`enabled`**: `true` or `false`
- **`mode`**: Dithering algorithm (see below)
- **`parameters`**: Algorithm-specific parameters (optional)

#### `palette`
- **`source`**: Palette generation method (see below)
- **`num_colors`**: Number of colors (ignored for custom palettes)
- **`use_gamma`**: `true` or `false` - Gamma correction

#### `final_resize`
- **`enabled`**: `true` or `false`
- **`multiplier`**: Integer multiplier for upscaling

---

## Dithering Algorithms

Available in the `dithering.mode` field:

| Mode | Description | Best For |
|------|-------------|----------|
| `none` | No dithering | Solid colors, testing |
| `bayer` | Ordered dithering | Fast, classic pixel art |
| `error_diffusion` | Floyd-Steinberg and variants | High quality, gradients |
| `blue_noise` | High-quality spatial | Smooth gradients |
| `IGN` | Interleaved gradient noise | Fine grain noise |
| `halftone` | Newspaper-style | Print effects |
| `riemersma` | Hilbert curve-based | Line art, details |
| `ostromoukhov` | Adaptive error diffusion | Natural images |
| `polka_dot` | Circular threshold patterns | Retro effects |
| `wavelet` | Multi-scale frequency | Technical images |
| `adaptive_variance` | Context-aware | Mixed content |
| `perceptual` | Luminance-preserving | Photos |
| `hybrid` | Luminance/color separation | High-quality photos |

---

## Palette Sources

Available in the `palette.source` field:

### Generated Palettes (use `num_colors`)

| Source | Description |
|--------|-------------|
| `"median_cut"` | Classic color quantization (fast) |
| `"kmeans"` | Machine learning clustering (better quality) |
| `"uniform"` | Evenly distributed color space |
| `"file:path.png"` | Extract from another image |

### Custom Palettes (ignore `num_colors`)

| Source | Description |
|--------|-------------|
| `"gameboy_dmg"` | Palette name from palette.json |
| `"custom:palette_name"` | Explicit custom palette reference |

**Note:** Custom palettes use their actual color count and ignore the `num_colors` setting.

---

## Processing Modes

### Single Image

```json
{
    "input": "photo.jpg",
    "output": "photo_dithered.png"
}
```

Automatically detected as image mode.

### Single Video

```json
{
    "input": "video.mp4",
    "output": "video_dithered.mp4"
}
```

Features:
- Preserves audio
- Shows progress bar
- Multi-core processing
- Even dimensions enforcement

### Batch Folder

```json
{
    "input": "images_folder",
    "output": "output_folder",
    "mode": "folder"
}
```

Features:
- Processes all images and videos
- Pre-loads neural models (huge performance boost)
- Continues on errors
- Summary statistics

---

## Usage Examples

### Basic Image Dithering

```bash
# Using config file
python dither_pie.py examples/image_basic.json

# Using config with input override
python dither_pie.py examples/image_basic.json my_photo.jpg
```

### Retro Pixel Art Effect

```bash
python dither_pie.py examples/image_pixelized.json
```

### Neural Pixelization

```bash
# Process single image
python dither_pie.py examples/image_neural.json

# Apply same neural settings to different file
python dither_pie.py examples/image_neural.json another_image.png
```

### Custom Palette

```bash
python dither_pie.py examples/image_custom_palette.json
```

### Video Processing

```bash
# Using config file
python dither_pie.py examples/video_basic.json

# Apply video settings to different video
python dither_pie.py examples/video_basic.json my_video.mp4
```

### Batch Processing

```bash
# Using config file
python dither_pie.py examples/batch_folder.json

# Apply same settings to different folder
python dither_pie.py examples/batch_folder.json vacation_photos/
# Output: vacation_photos_processed/
```

### Input Override Workflow

```bash
# 1. Create or edit a settings file with your preferred parameters
python dither_pie.py --example-config > my_style.json
# Edit my_style.json: set dithering mode, colors, pixelization, etc.

# 2. Apply to any file/folder
python dither_pie.py my_style.json photo1.jpg
python dither_pie.py my_style.json photo2.png
python dither_pie.py my_style.json video.mp4
python dither_pie.py my_style.json all_photos/

# Each generates appropriate output with descriptive filename
```

---

## Path Handling

All paths in the config file are:
1. Resolved relative to the config file location
2. Or used as absolute paths if specified

Example:
```
project/
  configs/
    process.json
  images/
    input.png
```

In `process.json`:
```json
{
    "input": "../images/input.png",
    "output": "../output/result.png"
}
```

---

## Tips & Best Practices

### Performance

- Use `"regular"` pixelization for speed
- Use `"neural"` pixelization for quality
- Enable verbose logging (`-v`) to diagnose issues
- Pre-load neural models once in batch mode

### Quality

- Enable gamma correction for accurate colors: `"use_gamma": true`
- Use `"error_diffusion"` or `"blue_noise"` for smooth gradients
- Try `"hybrid"` mode for high-quality photos
- Use larger `max_size` to preserve more detail

### Batch Processing

- Test on a single file first
- Use `"regular"` pixelization for large batches
- Watch for disk space (videos can be large)
- Use Ctrl+C to interrupt if needed

### Troubleshooting

- **Config validation errors**: Check field names and values
- **File not found**: Use absolute paths or check relative paths
- **Neural models fail**: Ensure `.pth` files are in project root
- **Video processing fails**: Check FFmpeg installation (`ffmpeg -version`)
- **Memory issues**: Reduce `max_size` or use `"regular"` pixelization

---

## Output Examples

### Success

```
░░░░       ░░░        ░░        ░░  ░░░░  ░░        ░░       ░░░░░░░░░       ░░░        ░░        ░░░░░░░░░      ░░░  ░░░░░░░░        ░░░░
▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒
▓▓▓▓  ▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓▓        ▓▓      ▓▓▓▓       ▓▓▓▓▓▓▓▓▓       ▓▓▓▓▓▓  ▓▓▓▓▓      ▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓
████  ████  █████  ████████  █████  ████  ██  ████████  ███  █████████  ███████████  █████  ██████████████  ████  ██  ███████████  ███████
████       ███        █████  █████  ████  ██        ██  ████  ████████  ████████        ██        █████████      ███        ██        ████


[18:30:00] INFO     Loading configuration from: config.json
           INFO     ✓ Configuration validated
           INFO     Auto-detected mode: image
           INFO     Input:  /path/to/input.png
           INFO     Output: /path/to/output.png
           INFO     Mode:   image
           INFO     Pixelization: regular (max_size=64)
           INFO     Dithering: bayer
           INFO     Palette: median_cut (16 colors)
           
           INFO     Loading image: input.png
           INFO     Image size: 512x512
           INFO     Pixelizing (regular, max_size=64)...
           INFO     ✓ Pixelized to 64x64
           INFO     Applying dithering: bayer
           INFO     Generating palette: median_cut (16 colors)
           INFO     ✓ Palette ready with 16 colors
           INFO     ✓ Dithering complete
           INFO     Saving to: /path/to/output.png
           INFO     ✓ Image saved successfully! (5.2 KB)
           
           INFO     ✓ Processing complete!
```

### Batch Summary

```
═══════════════════════════════════════
Batch Processing Summary
═══════════════════════════════════════
Total files:     10
Successful:     8
Failed:         2

Failed files:
  • corrupted_image.png
  • broken_video.mp4
```

---

## Integration with GUI

You can use both GUI and CLI modes:

```bash
# Launch GUI (no arguments)
python dither_pie.py

# Launch CLI (with arguments)
python dither_pie.py config.json
```

The GUI and CLI use the same underlying dithering library, so results are identical.

---

## Advanced Usage

### Custom Dithering Parameters

Some algorithms support parameters (e.g., Bayer matrix selection):

```json
{
    "dithering": {
        "mode": "bayer",
        "parameters": {
            "size": "psx4x4"
        }
    }
}
```

See `dithering_lib.py` for available parameters per algorithm.

IGN parameters example:
```json
{
    "dithering": {
        "mode": "IGN",
        "parameters": {
            "scale": 1.0,
            "seed": 0
        }
    }
}
```

### Environment Variables

- Set `PYTHONUNBUFFERED=1` for real-time output in CI/CD

### Automation

Use in scripts or batch files:

```bash
#!/bin/bash
for img in images/*.png; do
    python dither_pie.py configs/process.json --input "$img" --output "output/$img"
done
```

---

## See Also

- `README_NEW.md` - Full feature documentation
- `examples/` - Example configurations
- `CLI_IMPLEMENTATION_PLAN.md` - Development notes

---

**For issues or questions, see the main README or project repository.**

