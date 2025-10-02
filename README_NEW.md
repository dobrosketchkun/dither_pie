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

### 11 Dithering Algorithms
- **Bayer Matrices** (2x2, 4x4, 8x8, 16x16) - Ordered dithering patterns
- **Riemersma** - Hilbert curve-based error diffusion
- **Blue Noise** - High-quality spatial distribution
- **Polka Dot** - Circular threshold patterns
- **Wavelet** - Multi-scale frequency decomposition
- **Adaptive Variance** - Context-aware dithering
- **Perceptual** - Luminance-preserving error diffusion
- **Hybrid** - Separates luminance/color for detail preservation

### Advanced Features
- 🚀 **Smart Caching**: Re-dither without re-pixelizing
- 👁️ **Live Palette Preview**: See results in main window before applying
- ⚡ **Skip Pixelization**: Apply dithering directly to full-resolution images
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
> - Bayer 4x4
> - Riemersma
> - Blue Noise
> - Hybrid
> - Wavelet
> - Perceptual
> - All using same palette (16 colors)
> - 2x3 or 3x2 grid layout
> - Labels showing mode name

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
3. Apply Dithering - Bayer4x4, 16 colors
4. Enable "Upscale to original size"
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

---

## ⚡ Performance Features

### Smart Pixelization Cache
> **TODO: Diagram or flowchart showing:**
> - User pixelizes image → Result cached
> - User tries different dithering → Uses cache (instant)
> - User changes max_size → Cache invalidated, re-pixelize
> - Arrows showing "Cache Hit" vs "Cache Miss" paths

![Cache System](path/to/diagram_cache.png)

**Benefits:**
- Neural pixelization: 5-30 seconds saved per cache hit
- Experiment with dithering settings instantly
- No manual management needed

### Multi-core Video Processing
- Automatic worker pool (up to 4 cores)
- Batch processing for efficiency
- Progress tracking in real-time
- Audio/subtitle preservation

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
    "dither_mode": "bayer4x4",
    "use_gamma": false
  },
  "paths": {
    "last_image_dir": "C:/Users/...",
    "last_video_dir": "C:/Videos/..."
  },
  "recent_files": [...]
}
```

**What's Remembered:**
- ✅ Window size and position
- ✅ All processing defaults
- ✅ Last used directories
- ✅ Theme preferences
- ✅ Recent files list

---

## 🎥 Video Processing Details

### Features
- Frame-by-frame processing with dithering
- Audio stream preserved
- Subtitle track support
- H.264 encoding with yuv420p
- Even dimension enforcement (codec compatibility)

### Performance
> **TODO: Table or chart showing:**
> - Video length vs processing time
> - Different settings (regular vs neural, resolution)
> - Example: "1080p 30fps 1min video: ~2 min (regular), ~15 min (neural)"

| Video | Settings | Processing Time |
|-------|----------|----------------|
| 1080p 30fps 60sec | Regular + Bayer | ~2 minutes |
| 1080p 30fps 60sec | Neural + Blue Noise | ~15 minutes |
| 720p 24fps 30sec | Regular + Riemersma | ~45 seconds |

### Video Workflow
1. Load video → Shows first frame
2. Test on random frames
3. Configure settings with live preview
4. "Apply to Video" → Choose output path
5. Multi-core processing with progress bar

---

## 💡 Tips & Best Practices

### For Best Quality
- ✅ Use **Neural pixelization** for organic subjects (faces, nature)
- ✅ Use **Regular pixelization** for geometric/UI elements
- ✅ Enable **Gamma correction** for accurate color perception
- ✅ Try **Blue Noise** or **Hybrid** dithering for smooth gradients
- ✅ Use **Riemersma** for detailed line art

### Performance Tips
- ✅ Test on random frames before processing full video
- ✅ Use lower max_size for faster neural processing
- ✅ Regular pixelization is 50-100x faster than neural
- ✅ Preview different palettes without re-pixelizing (uses cache)

### Common Use Cases
- **Print/Halftone Effects**: Dither only (no pixelization), high color count
- **8-bit Game Art**: Regular + Bayer 4x4, 4-16 colors, custom palette
- **Artistic Effects**: Neural + Blue Noise, 32-64 colors, gamma on
- **Web Optimization**: Dither + resize for smaller file sizes

---

## 🔧 Technical Details

### Architecture
```
dither_pie.py          → Main GUI (customtkinter)
dithering_lib.py       → 11 dithering algorithms (strategy pattern)
video_processor.py     → Multi-core video processing
config_manager.py      → Persistent configuration
gui_components.py      → Reusable UI widgets
utils.py              → Palette management
models/               → Neural pixelization models
```

### Dependencies
- **GUI**: customtkinter, tkinter
- **Image**: Pillow, numpy
- **ML**: PyTorch, scikit-learn
- **Video**: FFmpeg (subprocess), opencv-python
- **Math**: scipy, pywavelets

### Credits
Neural pixelization models from:
- [arenasys/pixelization_inference](https://github.com/arenasys/pixelization_inference)
- [WuZongWei6/Pixelization](https://github.com/WuZongWei6/Pixelization)

---

## 📝 Changelog

### Latest Features
- ✨ Live palette preview in main window
- ✨ Smart pixelization caching
- ✨ Dithering without pixelization
- ✨ Persistent user preferences (config.json)
- ✨ Recent files tracking
- ✨ Last used directory memory

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

