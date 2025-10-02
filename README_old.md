# Image Dithering Tool

This repository provides a Python-based tool for applying different forms of `dithering` to images **and** videos via a **Graphical User Interface (GUI)**. In addition to traditional pixelization using nearest-neighbor interpolation, the tool supports neural network–based pixelization for a more refined low-resolution effect. It also offers multiple dithering modes and custom palette management.

**Performance optimized:** Uses multiprocessing for fast video processing, model caching for neural pixelization, and threading to keep the GUI responsive.

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

## Quick Start

1. **Install Dependencies** (see above)

2. **Run the Application**  
   `python dither_pie.py`

   This launches the GUI application. From there you can:
   - Open images or videos
   - Choose pixelization: regular (nearest-neighbor) or neural
   - Select dithering modes (e.g. `bayer4x4`, `riemersma`, `wavelet`, `blue_noise`, etc.)
   - Adjust color count and enable gamma correction
   - Save your results

---

## GUI Screenshots

Below are a few example screenshots illustrating the interface:

[<img src="misc/Screenshot_main_window.jpg" alt="Main Window Screenshot" />](## "Screenshot of the main window")  
[<img src="misc/Screenshot_palette_dialog.jpg" alt="Palette Dialog Screenshot" />](## "Screenshot of the palette dialog")

---

## Neural Pixelization

This tool includes a neural network–based pixelization mode that produces a refined, artistic pixelized effect compared to traditional nearest-neighbor interpolation. To use it:
- Click the "Pixelize (Neural)" button in the GUI

Ensure you have downloaded the necessary model files from the [link](https://mega.nz/folder/mdtnQT4K#ZkoSrVAIubonAzZuJ9QUlA).


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

You can edit palettes by manually editing `palette.json`.


---

## Additional Notes

- The tool enforces even dimensions after pixelization (crucial for video encoding with codecs like libx264)
- Original repositories for neural pixelization:  
   - https://github.com/arenasys/pixelization_inference  
   - https://github.com/WuZongWei6/Pixelization


