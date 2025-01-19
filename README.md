# Image Dithering Tool

This repository provides a Python-based tool for applying different forms of **dithering** to images, either through a **Graphical User Interface (GUI)** or via a **Command-Line Interface (CLI)**. The tool also supports **pixelization** (reducing image size with nearest-neighbor interpolation) and allows the creation and management of **custom palettes**.

Below is an overview of the major modes:

- **GUI Application**: Run the application in a window, load images, pixelize them, select a dithering mode, choose or create palettes, and preview results interactively.
- **CLI Tool**: Perform the same operations (pixelize, dither, or both) via command-line commands, which is particularly useful for automation or scripting.


---

## Quick Start (GUI)

1. **Install Dependencies**  
   - Make sure you have Python 3.7+ installed, plus `Pillow`, `numpy`, `scikit-learn`, `customtkinter`, and `tkinter`.
2. **Run the GUI**  
   - `python dither_pie.py`  
   This launches the main window. From there, you can open an image, choose dithering/pixelization options, and save the result.

### GUI Screenshots

Here are a couple of screenshots illustrating the interface:
  
[<img src="/misc/Screenshot_main_window.jpg" />](## "Screenshot of the main window")
[<img src="/misc/Screenshot_palette_dialog.jpg" />](## "Screenshot of the palette dialog")
---

## Using the CLI

For command-line usage, run `python dither_pie.py [command] [options]`. The available commands are:

1. **pixelize**  
   - Usage: `python dither_pie.py pixelize input_image output_image [--max-size N]`
   - Resizes the image with nearest neighbor to create a pixelized look.

2. **dither**  
   - Usage: `python dither_pie.py dither input_image output_image [--mode bayer4x4] [--colors 16] [--algo-palette kmeans_variant1] [--palette customName]`
   - Applies dithering to an input image. You can specify:
     - `--mode` to choose a dithering matrix (e.g., `bayer2x2`, `bayer4x4`, `bayer8x8`, etc.).
     - `--colors` to set the total number of colors.
     - `--algo-palette` to select a built-in palette generation method (`median_cut`, `kmeans_variant1`, etc.).
     - `--palette` to use a custom palette from the `palette.json` file.

3. **dither-pixelize**  
   - Usage: `python dither_pie.py dither-pixelize input_image output_image [options...]`
   - Combines pixelization and dithering in a single step.

---

## Example Dithered Images

Below are some selected images demonstrating different modes and palettes:

[<img src="/misc/grid_image.png" />](## "7cats / Roberta")

[<img src="/misc/grid_image2.png" />](## "ink (inksgirls)")

[<img src="/misc/grid_image3.png" />](## "asagi ryo")

[<img src="/misc/grid_image4.png" />](## "kenomotsu yukuwa / MiSide")


---

## Custom Palettes

- The file `palette.json` contains user-defined palettes. Each entry has a `name` and a list of `colors` in hex format.
- You can create new palettes either via the GUI (Palette Dialog) or by editing `palette.json` manually.

Example of one palette:

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
```
