# Image Dithering Tool

This repository provides a Python-based tool for applying different forms of `dithering` to images **and** videos via both a **Graphical User Interface (GUI)** and a **Command-Line Interface (CLI)**. In addition to traditional pixelization using nearest-neighbor interpolation, the tool now supports neural network–based pixelization for a more refined low-resolution effect. It also offers multiple dithering modes and custom palette management.

All neural pixelization model files are available [here](https://mega.nz/folder/mdtnQT4K#ZkoSrVAIubonAzZuJ9QUlA).  


---

## Install Dependencies

Ensure you have Python 3.7+ installed. Then install the following packages:

- `Pillow`
- `numpy`
- `scikit-learn`
- `customtkinter`
- `cv2`
- `tkinter` (usually included with Python)
- `pywt` (for wavelet-based dithering)
- `torch` and `torchvision` (for neural pixelization)
- `ffmpeg` (for video processing; ensure it is in your system `PATH`)

---

## Quick Start (GUI)

1. **Install Dependencies** (see above)

2. **Run the GUI**  
   `python dither_pie.py`

   This launches the main application window. From there you can:
   - Open images or videos.
   - Choose pixelization options: traditional (nearest-neighbor) or neural (check the `--neural` flag or option).
   - Select various dithering modes (e.g. `bayer4x4`, `floydsteinberg`, `wavelet`, `polka_dot`, etc.).
   - Manage and apply custom palettes.
   - Enable `Gamma Correction` if needed.

---

## GUI Screenshots

Below are a few example screenshots illustrating the interface:

[<img src="misc/Screenshot_main_window.jpg" alt="Main Window Screenshot" />](## "Screenshot of the main window")  
[<img src="misc/Screenshot_palette_dialog.jpg" alt="Palette Dialog Screenshot" />](## "Screenshot of the palette dialog")

---

## Using the CLI

Run commands in the following format:  
`python dither_pie.py [command] [options]`

### 1. pixelize

Pixelizes an image or video using nearest-neighbor interpolation or neural pixelization (add the `--neural` flag).

**Usage:**  
`python dither_pie.py pixelize input_path output_path --max-size [size] [--neural]`

**Example:**  
`python dither_pie.py pixelize video.mp4 output.mp4 --max-size 640 --neural`

---

### 2. dither

Applies dithering to an image or video.

**Usage:**  
`python dither_pie.py dither input_path output_path --mode [dither_mode] --colors [number] [--algo-palette palette_name] [--palette custom_palette_name] [--gamma-correction]`

**Example:**  
`python dither_pie.py dither image.jpg output.png --mode bayer4x4 --colors 16 --algo-palette median_cut`

---

### 3. dither-pixelize

Performs pixelization and dithering in one step (ideal for videos). Supports both nearest-neighbor and neural pixelization (use `--neural`).

**Usage:**  
`python dither_pie.py dither-pixelize input_path output_path --max-size [size] [--neural] --mode [dither_mode] --colors [number] [--algo-palette palette_name] [--palette custom_palette_name] [--gamma-correction] [--final-resize final_size]`

**Example:**  
`python dither_pie.py dither-pixelize video.mp4 output.mp4 --max-size 320 --neural --mode bayer4x4 --colors 16 --final-resize 800`

---

### 4. import-lospal

Imports a palette from lospec.com and adds it to `palette.json`.

**Usage:**  
`python dither_pie.py import-lospal palette_url`

**Example:**  
`python dither_pie.py import-lospal https://lospec.com/palette-list/my-cool-palette`

---

### 5. create-pal-from-image

Generates a new palette using K-means clustering from an image and saves it to `palette.json`.

**Usage:**  
`python dither_pie.py create-pal-from-image input_image --colors [number] --name palette_name`

**Example:**  
`python dither_pie.py create-pal-from-image image.jpg --colors 16 --name MyPalette`

---

### 6. resize

Upscales or downscales an image or video using nearest-neighbor interpolation. The tool enforces even dimensions (critical for video codecs like libx264).

**Usage:**  
`python dither_pie.py resize target_size input_path`

**Example:**  
`python dither_pie.py resize 640 video.mp4`

---

## Neural Pixelization

This tool includes a neural network–based pixelization mode that produces a refined, artistic pixelized effect compared to traditional nearest-neighbor interpolation. To use neural pixelization:
- In the GUI, use the neural pixelize button.
- In the CLI, add the `--neural` flag to the `pixelize` or `dither-pixelize` commands.


Ensure you have downloaded the necessary model files from the [link](https://mega.nz/folder/mdtnQT4K#ZkoSrVAIubonAzZuJ9QUlA). The neural approach also enforces even dimensions to avoid encoding issues.


---

## Example Dithered Images

Below are examples of still images processed with various dithering modes and palettes:

[<img src="misc/grid_image.png" alt="Dithered Sample" />](## "7cats / Roberta")  
[<img src="misc/grid_image2.png" alt="Dithered Sample 2" />](## "ink / inksgirls")  
[<img src="misc/grid_image3.png" alt="Dithered Sample 3" />](## "asagi ryo sample")  
[<img src="misc/grid_image4.png" alt="Dithered Sample 4" />](## "kenomotsu yukuwa / MiSide")

---

## Video Previews

The `misc` folder includes short video demos illustrating the results of pixelization and dithering on sample clips. You can watch them below:

<video src="https://github.com/user-attachments/assets/51eeacd6-c3bb-4e4e-9249-bcc954649c68"></video>  
<video src="https://github.com/user-attachments/assets/f81f9e8a-4cb4-4acf-8619-96709243e227"></video>  
<video src="https://github.com/user-attachments/assets/8ba5a42c-407b-48f5-95ea-6cf6d941b219"></video>

These clips demonstrate how pixelization and different dithering modes combine to produce retro or stylized results.

---

## Custom Palettes

A `palette.json` file stores user-defined palettes. Each entry has a `name` and a list of `colors` (hex codes). For example:

```json
{
  "name": "gb_dmg_palette",
  "colors": [
    "#0f381f",
    "#304e2a",
    "#8bac0f",
    "#9bce0f"
  ]
}
````

You can create, import, or edit palettes via:
- **GUI**: Click `Apply Dithering` → `Select Palette` → `Create Custom Palette` or `Import from lospec.com`.
- **CLI**: Use `import-lospal` or `create-pal-from-image`, or manually edit `palette.json`.


---

## Additional Notes

- The tool enforces even dimensions after pixelization (crucial for video encoding with codecs like libx264).
- Both the GUI and CLI fully support neural pixelization along with all existing dithering options.
- Original repositories for neural pixelization:  
   - https://github.com/arenasys/pixelization_inference  
   - https://github.com/WuZongWei6/Pixelization


