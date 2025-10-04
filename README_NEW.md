# Dither Pie - Advanced Image & Video Dithering Tool

A powerful Python-based GUI application for applying artistic dithering and pixelization effects to images and videos. Features intelligent caching, live previews, and professional-quality retro visual effects.

---

## 🎯 Key Features

### Image & Video Processing
- ✅ **Images**: PNG, JPG, JPEG, GIF, BMP support
- ✅ **Videos**: MP4, AVI, MOV, MKV with audio preservation
- ✅ **Random Frame Preview**: Test settings on any video frame
- ✅ **Multi-core Processing**: Fast video rendering with multiprocessing

### Pixelization Methods
- **Regular**: Fast nearest-neighbor downsampling for classic pixel art
- **Neural**: AI-powered artistic pixelization using PyTorch models

### 12 Dithering Algorithms
- **Bayer** - Ordered dithering with configurable matrix size (2x2, 4x4, 8x8, 16x16)
- **Error Diffusion** - 8 classic algorithms in one: Floyd-Steinberg, JJN, Stucki, Burkes, Atkinson, Sierra (3 variants)
- **Ostromoukhov** - Adaptive error diffusion with variable coefficients
- **Riemersma** - Hilbert curve-based space-filling error diffusion
- **Blue Noise** - High-quality spatial distribution with configurable seed
- **Polka Dot** - Retro circular threshold patterns
- **Halftone** - Newspaper-style printing simulation with rotating screens
- **Wavelet** - Multi-scale frequency decomposition dithering
- **Adaptive Variance** - Context-aware dithering based on local variance
- **Perceptual** - Luminance-preserving error diffusion
- **Hybrid** - Separates luminance/color channels for detail preservation

### Advanced Features
- 🚀 **Smart Caching**: Re-dither without re-pixelizing, blue noise matrices cached in memory
- 👁️ **Live Palette Preview**: See results in main window before applying
- 🔄 **Toggle View During Preview**: Compare original with preview while selecting palette
- ⚙️ **Configurable Algorithms**: Fine-tune parameters for 10+ dithering modes via settings dialog
- 🌀 **Animated Status Bar**: Visual feedback with customizable spinner animations
- 💾 **Persistent Settings**: Window position, defaults, and paths remembered
- 🎨 **Custom Palettes**: Create, import from lospec.com, or extract from images

---

## 📸 Screenshots

### Main Interface
> **TODO: Screenshot of main window showing:**
> - Left sidebar with all controls (pixelization settings, dithering options, color count)
> - Main viewer showing a dithered image (use something colorful - landscape or character art)
> - Status bar at bottom
> - Window should show realistic size (1400x900)

![Main Window](path/to/screenshot_main.png)

### Live Palette Preview
> **TODO: Side-by-side comparison showing:**
> - Left: Palette selection dialog (400x600) with radio buttons and color bars
> - Right: Main window displaying the live preview at full size
> - Arrow or annotation showing "Preview updates here instantly"
> - Show 3-4 different palettes in the list (Median Cut, K-means, custom palette, etc.)

![Live Preview Feature](path/to/screenshot_live_preview.png)

### Before/After Comparison Grid
> **TODO: 2x2 or 3x3 grid showing:**
> - Original image (top-left)
> - Regular pixelization only (top-right)
> - Neural pixelization only (bottom-left)
> - Neural pixelization + dithering with custom palette (bottom-right)
> - Use the same source image for all variations
> - Labels on each image

![Processing Comparison](path/to/screenshot_comparison.png)

### Dithering Modes Showcase
> **TODO: Grid showing the same image with different dithering algorithms:**
> - Bayer (4x4)
> - Error Diffusion (Floyd-Steinberg)
> - Error Diffusion (Atkinson)
> - Blue Noise
> - Halftone
> - Hybrid
> - All using same palette (16 colors)
> - 2x3 grid layout
> - Labels showing mode name and variant

![Dithering Algorithms](path/to/screenshot_dither_modes.png)

### Custom Palette Dialog
> **TODO: Screenshot showing:**
> - HSV color picker dialog with hue gradient bar
> - Saturation-Value plane with selection circle
> - RGB/HEX input fields
> - Preview of selected color
> - Show it in action (mid-use, not empty)

![Custom Palette Creator](path/to/screenshot_palette_creator.png)

### Video Processing
> **TODO: Animated GIF or video showing:**
> - Original video clip (3-5 seconds)
> - Processed version with pixelization + dithering
> - Side-by-side or before/after format
> - Progress indicator visible if possible
> - Use something with motion (person walking, car driving, etc.)

![Video Processing Demo](path/to/video_demo.gif)

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- FFmpeg (in system PATH for video processing)

### Required Packages
```bash
pip install pillow numpy scikit-learn customtkinter opencv-python pywavelets torch torchvision
```

### Neural Models (Optional)
Download model files for neural pixelization from:
[Neural Models - MEGA](https://mega.nz/folder/mdtnQT4K#ZkoSrVAIubonAzZuJ9QUlA)

Place in project root:
- `160_net_G_A.pth`
- `alias_net.pth`
- `pixelart_vgg19.pth`

---

## 📖 Quick Start Guide

### Basic Workflow
1. **Load Image/Video** → Click "Load Image" or "Load Video"
2. **Pixelize** (optional) → Choose Regular or Neural, set max size
3. **Apply Dithering** → Select palette and see live preview
4. **Save Result** → Export as PNG or process full video

### Example Workflows

#### Classic Pixel Art Effect
```
1. Load Image
2. Pixelize (Regular) - Max Size: 64
3. Apply Dithering - Bayer (4x4), 16 colors
4. Enable "Final resize"
5. Save Result
```

#### High-Quality Retro Effect
```
1. Load Image
2. Pixelize (Neural) - Max Size: 128
3. Apply Dithering - Blue Noise, 32 colors, Gamma Correction ON
4. Try different palettes in live preview
5. Apply Selected → Save
```

#### Dithering Only (No Pixelization)
```
1. Load Image
2. Skip pixelization step
3. Apply Dithering - Hybrid mode, extract palette from another image
4. Save Result (preserves full resolution)
```

---

## 🎨 Palette Management

### Built-in Palette Generation
- **Median Cut**: Classic color quantization algorithm
- **K-means**: Machine learning-based clustering
- **Uniform**: Evenly distributed color space

### Custom Palettes
1. **Manual Creation**: HSV color picker with RGB/HEX input
2. **Import from Lospec.com**: Paste URL, instant import
3. **Extract from Image**: K-means clustering on reference image

### Palette File
Palettes stored in `palette.json`:
```json
{
  "name": "gameboy_dmg",
  "colors": ["#0f381f", "#304e2a", "#8bac0f", "#9bce0f"]
}
```

### Live Preview Workflow
During palette selection, you can:
- **View Live Previews**: Each palette generates a preview shown in the main window
- **Toggle View**: Click "Toggle View" button to compare original image with current preview
- **Adjust Settings**: Change dither mode or parameters (⚙️) while previewing
- **Zoom & Pan**: Examine details - zoom/pan state preserved between palette switches
- **Toggle Gamma**: See instant difference with/without gamma correction

This allows you to:
1. Select a palette → Preview appears
2. Toggle to original → Compare side-by-side mentally
3. Try another palette → Toggle back to see new preview
4. Adjust parameters → Preview updates automatically
5. Apply when satisfied

---


## 🎛️ Configuration System

User preferences automatically saved to `config.json`:

```json
{
  "window": {
    "width": 1400,
    "height": 900,
    "x": 100,
    "y": 50,
    "maximized": false
  },
  "defaults": {
    "max_size": 640,
    "num_colors": 16,
    "dither_mode": "bayer",
    "use_gamma": false
  },
  "ui": {
    "spinner_name": "dots"
  },
  "paths": {
    "last_image_dir": "C:/Users/...",
    "last_video_dir": "C:/Videos/..."
  },
  "recent_files": [...]
}
```


---

## 🎥 Video Processing Details

### Features
- Frame-by-frame processing with dithering
- Audio stream preserved
- Subtitle track support
- H.264 encoding with yuv420p
- Even dimension enforcement (codec compatibility)


### Video Workflow
1. Load video → Shows first frame
2. Test on random frames
3. Configure settings with live preview
4. "Apply to Video" → Choose output path
5. Multi-core processing with progress bar

---

## ⚙️ Configurable Dithering Parameters

Many dithering modes include adjustable parameters accessible via the **⚙️ settings button** next to the dithering mode dropdown.

### Algorithms with Settings

| Algorithm | Configurable Parameters | Description |
|-----------|------------------------|-------------|
| **Bayer** | Matrix Size | Choose 2x2, 4x4, 8x8, or 16x16 (default: 4x4) |
| **Error Diffusion** | Variant, Serpentine Scan | 8 algorithms: Floyd-Steinberg, JJN, Stucki, Burkes, Atkinson, Sierra (3 variants). Toggle serpentine scanning for artifact reduction. |
| **Ostromoukhov** | Serpentine Scan | Adaptive error diffusion with optional serpentine scanning |
| **Blue Noise** | Matrix Size, Random Seed | Size (64-512), seed for reproducible patterns |
| **Polka Dot** | Tile Size, Gamma | Dot pattern size and gamma adjustment |
| **Halftone** | Cell Size, Screen Angle, Dot Gain, Dot Size Range, Shape, Sharpness | Full control over newspaper-style halftone printing |
| **Wavelet** | Wavelet Type, Subband Quantization | Choose wavelet family (haar, db1-10) and quantization strength |
| **Adaptive Variance** | Variance Threshold, Window Radius | Context-awareness sensitivity and local analysis window |
| **Hybrid** | Luminance Factor, Color Factor | Balance between luminance and color channel processing |

### How to Use Settings
1. Select a dithering mode from dropdown
2. Click **⚙️** button (enabled for modes with parameters)
3. Adjust parameters in dialog
4. Click **Apply** to update preview instantly
5. Changes are cached with each palette for easy comparison

**Note:** Settings button is disabled for modes without configurable parameters (None, Riemersma, Perceptual).

---

## 💡 Tips & Best Practices

### For Best Quality
- ✅ Use **Neural pixelization** for organic subjects (faces, nature)
- ✅ Use **Regular pixelization** for geometric/UI elements
- ✅ Enable **Gamma correction** for accurate color perception
- ✅ Try **Blue Noise**, **Error Diffusion (Atkinson)**, or **Hybrid** for smooth gradients
- ✅ Use **Halftone** for authentic newspaper/magazine printing effects
- ✅ Use **Riemersma** or **Error Diffusion (Floyd-Steinberg)** for detailed line art
- ✅ Adjust algorithm parameters via ⚙️ settings button for fine control

### Performance Tips
- ✅ Test on random frames before processing full video
- ✅ Use lower max_size for faster neural processing
- ✅ Regular pixelization is 50-100x faster than neural
- ✅ Preview different palettes without re-pixelizing (uses cache)

### Common Use Cases
- **Print/Halftone Effects**: Halftone mode (no pixelization), adjust screen angle and dot size
- **8-bit Game Art**: Regular + Bayer (4x4), 4-16 colors, custom palette
- **Retro Mac Look**: Error Diffusion (Atkinson variant), serpentine off
- **Artistic Effects**: Neural + Blue Noise or Error Diffusion (JJN), 32-64 colors, gamma on
- **Newspaper Style**: Halftone mode with 6-8 colors, angle 45°, gamma on
- **Web Optimization**: Dither + resize for smaller file sizes with preserved detail

---

## 🔧 Technical Details

### Architecture
```
dither_pie.py          → Main GUI (customtkinter) with live preview system
dithering_lib.py       → 12 dithering algorithms with configurable parameters
video_processor.py     → Multi-core video processing with FFmpeg
config_manager.py      → Persistent configuration with JSON storage
gui_components.py      → Reusable UI widgets (settings dialog, animated status bar)
utils.py              → Palette management (generation, import, extraction)
models/               → Neural pixelization models (PyTorch)
spinners.json         → Animated spinner definitions for status bar
```

### Dependencies
- **GUI**: customtkinter, tkinter
- **Image**: Pillow, numpy
- **ML**: PyTorch, scikit-learn
- **Video**: FFmpeg (subprocess), opencv-python
- **Math**: scipy, pywavelets

### Technical Highlights
- **Strategy Pattern**: Each dithering algorithm is a separate strategy class
- **Metadata-Driven UI**: Parameter dialogs generated from algorithm metadata (`get_parameter_info()`)
- **Separation of Concerns**: GUI and core algorithms are fully decoupled
- **Smart Caching**: Multi-level caching (pixelization, preview, blue noise matrices)
- **Live Preview System**: Non-blocking preview generation with threading
- **Animated Feedback**: Status bar with configurable spinner animations from `spinners.json`
- **State Management**: Palette dialog state tracking for toggle view functionality

### Key Design Decisions
- **No Parameter Persistence Across Modes**: Each algorithm starts with defaults when selected
- **Preview Cache by Settings**: Cache key includes palette, gamma, dither mode, and all parameters
- **In-Memory Blue Noise**: Generated matrices cached during session, not persisted
- **Serpentine Off by Default**: Cleaner look for most use cases, easily toggled in settings

### Credits
Neural pixelization models from:
- [arenasys/pixelization_inference](https://github.com/arenasys/pixelization_inference)
- [WuZongWei6/Pixelization](https://github.com/WuZongWei6/Pixelization)

---

## 📝 Changelog

### Latest Features
- ✨ **Configurable Dithering Algorithms**: Settings dialog (⚙️) for fine-tuning 10+ modes
- ✨ **Consolidated Algorithms**: Bayer sizes and Error Diffusion variants in single modes
- ✨ **Toggle View During Preview**: Compare original with dithered preview in real-time
- ✨ **Animated Status Bar**: Visual feedback with customizable spinner styles
- ✨ **Blue Noise Caching**: Generated matrices cached in memory for performance
- ✨ **Halftone Mode**: Authentic newspaper printing simulation with configurable parameters
- ✨ **Live Palette Preview**: See results in main window before applying
- ✨ **Smart Pixelization Caching**: Instant re-dithering without re-processing
- ✨ **Persistent User Preferences**: All settings saved to config.json

### Coming Soon
- 🔜 Batch image processing
- 🔜 Preset management (save/load settings)
- 🔜 Export palette from result
- 🔜 Undo/redo system

---

## 📄 License

> **TODO: Add your license information here**

---

## 🤝 Contributing

> **TODO: Add contribution guidelines if open source**

---

## 📧 Contact

> **TODO: Add contact information or GitHub repo link**

