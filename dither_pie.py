import json
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from enum import Enum
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import math
import os
import urllib.request  # For HTTP requests
from scipy.spatial import KDTree

# -------------------- Enumerations and Utilities --------------------

class DitherMode(Enum):
    NONE = "none"
    BAYER2x2 = "bayer2x2"
    BAYER4x4 = "bayer4x4"
    BAYER8x8 = "bayer8x8"
    BAYER16x16 = "bayer16x16"  # Newly added mode

class DitherUtils:
    BAYER2x2 = np.array([
        [0.25, 0.75],
        [1.0, 0.5]
    ])

    BAYER4x4 = np.array([
        [0.03125, 0.53125, 0.15625, 0.65625],
        [0.78125, 0.28125, 0.90625, 0.40625],
        [0.21875, 0.71875, 0.09375, 0.59375],
        [0.96875, 0.46875, 0.84375, 0.34375]
    ])

    BAYER8x8 = np.array([
        [0.015625, 0.515625, 0.140625, 0.640625, 0.046875, 0.546875, 0.171875, 0.671875],
        [0.765625, 0.265625, 0.890625, 0.390625, 0.796875, 0.296875, 0.921875, 0.421875],
        [0.203125, 0.703125, 0.078125, 0.578125, 0.234375, 0.734375, 0.109375, 0.609375],
        [0.953125, 0.453125, 0.828125, 0.328125, 0.984375, 0.484375, 0.84375, 0.34375],
        [0.0625, 0.5625, 0.1875, 0.6875, 0.03125, 0.53125, 0.15625, 0.65625],
        [0.8125, 0.3125, 0.9375, 0.4375, 0.78125, 0.28125, 0.90625, 0.40625],
        [0.25, 0.75, 0.125, 0.625, 0.21875, 0.71875, 0.09375, 0.59375],
        [1.0, 0.5, 0.875, 0.375, 0.96875, 0.46875, 0.84375, 0.34375]
    ])

    BAYER16x16 = np.array([
        [0.00390625, 0.50390625, 0.12890625, 0.62890625, 0.03125, 0.53125, 0.15625, 0.65625,
         0.046875, 0.546875, 0.171875, 0.671875, 0.01171875, 0.51171875, 0.13671875, 0.63671875],
        [0.76367188, 0.26367188, 0.88867188, 0.38867188, 0.796875, 0.296875, 0.921875, 0.421875,
         0.7421875, 0.2421875, 0.8671875, 0.3671875, 0.98046875, 0.48046875, 0.90625, 0.40625],
        [0.203125, 0.703125, 0.078125, 0.578125, 0.21875, 0.71875, 0.09375, 0.59375,
         0.1484375, 0.6484375, 0.0234375, 0.5234375, 0.109375, 0.609375, 0.234375, 0.734375],
        [0.9453125, 0.4453125, 0.8203125, 0.3203125, 0.9609375, 0.4609375, 0.8359375, 0.3359375,
         0.890625, 0.390625, 0.765625, 0.265625, 0.984375, 0.484375, 0.859375, 0.359375],
        [0.0625, 0.5625, 0.1875, 0.6875, 0.03125, 0.53125, 0.15625, 0.65625,
         0.1015625, 0.6015625, 0.2265625, 0.7265625, 0.046875, 0.546875, 0.171875, 0.671875],
        [0.8125, 0.3125, 0.9375, 0.4375, 0.78125, 0.28125, 0.90625, 0.40625,
         0.8515625, 0.3515625, 0.9765625, 0.4765625, 0.796875, 0.296875, 0.921875, 0.421875],
        [0.2421875, 0.7421875, 0.1171875, 0.6171875, 0.2578125, 0.7578125, 0.1328125, 0.6328125,
         0.1484375, 0.6484375, 0.0234375, 0.5234375, 0.109375, 0.609375, 0.234375, 0.734375],
        [0.98046875, 0.48046875, 0.8671875, 0.3671875, 0.9765625, 0.4765625, 0.8515625, 0.3515625,
         0.921875, 0.421875, 0.796875, 0.296875, 0.90625, 0.40625, 0.78125, 0.28125]
    ])

    @staticmethod
    def get_threshold_matrix(mode: DitherMode) -> np.ndarray:
        if mode == DitherMode.BAYER2x2:
            return DitherUtils.BAYER2x2
        elif mode == DitherMode.BAYER4x4:
            return DitherUtils.BAYER4x4
        elif mode == DitherMode.BAYER8x8:
            return DitherUtils.BAYER8x8
        elif mode == DitherMode.BAYER16x16:
            return DitherUtils.BAYER16x16
        elif mode == DitherMode.NONE:
            return np.ones((1, 1))
        else:
            raise ValueError(f"Unsupported Dither Mode: {mode}")


class ColorReducer:
    @staticmethod
    def get_color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between colors in RGB space"""
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    @staticmethod
    def find_dominant_channel(colors: List[Tuple[int, int, int]]) -> int:
        """Find the color channel with the largest range"""
        ranges = [0, 0, 0]
        for channel in range(3):
            min_val = min(color[channel] for color in colors)
            max_val = max(color[channel] for color in colors)
            ranges[channel] = max_val - min_val
        return ranges.index(max(ranges))

    @staticmethod
    def median_cut(colors: List[Tuple[int, int, int]], depth: int) -> List[Tuple[int, int, int]]:
        """Perform median cut color quantization"""
        if depth == 0 or len(colors) == 0:
            if len(colors) == 0:
                return [(0, 0, 0)]
            avg_color = tuple(int(sum(channel) / len(colors)) for channel in zip(*colors))
            return [avg_color]

        channel = ColorReducer.find_dominant_channel(colors)
        colors.sort(key=lambda x: x[channel])

        mid = len(colors) // 2
        return (ColorReducer.median_cut(colors[:mid], depth - 1) +
                ColorReducer.median_cut(colors[mid:], depth - 1))

    @staticmethod
    def reduce_colors(image: Image.Image, num_colors: int) -> List[Tuple[int, int, int]]:
        """Extract and reduce colors from image using median cut"""
        image = image.convert('RGB')
        colors = list(set(image.getdata()))
        if num_colors < 1:
            num_colors = 1  # Safe fallback
        depth = int(math.log2(num_colors)) if num_colors > 1 else 0
        return ColorReducer.median_cut(colors, depth)


class ImageDitherer:
    def __init__(self, num_colors: int = 16, dither_mode: DitherMode = DitherMode.BAYER4x4, palette: List[Tuple[int, int, int]] = None):
        self.num_colors = num_colors
        self.dither_mode = dither_mode
        self.palette = palette
        self.threshold_matrix = DitherUtils.get_threshold_matrix(dither_mode)

    def apply_dithering(self, image: Image.Image) -> Image.Image:
        """Apply dithering to the image using KD-Tree + vectorized approach."""
        # 1) Generate palette if not provided
        if self.palette is None:
            self.palette = ColorReducer.reduce_colors(image, self.num_colors)

        # Convert image to array
        img_rgb = image.convert('RGB')
        arr = np.array(img_rgb, dtype=np.uint8)
        h, w, _ = arr.shape

        # Build KDTree for the palette (convert to float32)
        palette_arr = np.array(self.palette, dtype=np.float32)
        tree = KDTree(palette_arr)

        # Flatten pixels to shape (N,3)
        flat_pixels = arr.reshape((-1, 3)).astype(np.float32)

        if self.dither_mode == DitherMode.NONE:
            # -------------------- Handling DitherMode.NONE --------------------
            _, idx = tree.query(flat_pixels, k=1, workers=-1)
            dithered_pixels = palette_arr[idx, :]
            out_arr = dithered_pixels.reshape((h, w, 3)).astype(np.uint8)
            result = Image.fromarray(out_arr, mode='RGB')
            return result
        else:
            # -------------------- Handling Dither Modes with Threshold --------------------
            distances, indices = tree.query(flat_pixels, k=2, workers=-1)
            distances_sq = distances ** 2

            dist_nearest = distances_sq[:, 0]
            dist_second = distances_sq[:, 1]

            total_dist = dist_nearest + dist_second
            factor = np.where(total_dist == 0, 0.0, dist_nearest / total_dist)

            # Tile the threshold matrix to match image dimensions
            th_mat = self.threshold_matrix
            th_h, th_w = th_mat.shape
            tiled_threshold = np.tile(th_mat, ((h + th_h - 1) // th_h, (w + th_w - 1) // th_w))
            tiled_threshold = tiled_threshold[:h, :w]
            flat_threshold = tiled_threshold.flatten()

            idx_nearest = indices[:, 0]
            idx_second = indices[:, 1]
            use_nearest = (factor <= flat_threshold)
            final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)

            dithered_pixels = palette_arr[final_indices, :]
            out_arr = dithered_pixels.reshape((h, w, 3)).astype(np.uint8)
            result = Image.fromarray(out_arr, mode='RGB')
            return result

# -------------------- Image Viewer --------------------

class ZoomableImage(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.original_image: Optional[Image.Image] = None
        self.displayed_image: Optional[ImageTk.PhotoImage] = None
        self.image_id: Optional[int] = None

        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.offset_x = 0
        self.offset_y = 0

        # Bind events
        self.bind('<ButtonPress-1>', self.start_pan)
        self.bind('<B1-Motion>', self.pan)
        self.bind('<MouseWheel>', self.zoom)  # Windows
        self.bind('<Button-4>', self.zoom)    # Linux scroll up
        self.bind('<Button-5>', self.zoom)    # Linux scroll down

        # Bind resize
        self.bind('<Configure>', self.on_resize)

    def set_image(self, image: Image.Image):
        """Set a new image and reset view parameters"""
        self.original_image = image
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_view()

    def fit_to_window(self):
        """Scale image to fit window while maintaining aspect ratio"""
        if not self.original_image:
            return

        self.update_idletasks()
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        image_width, image_height = self.original_image.size

        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        self.zoom_factor = min(width_ratio, height_ratio)

        self.offset_x = 0
        self.offset_y = 0
        self.update_view()

    def update_view(self):
        """Update the displayed image based on current zoom and pan"""
        if not self.original_image:
            return

        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)

        if new_width <= 0 or new_height <= 0:
            return

        resized = self.original_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        self.displayed_image = ImageTk.PhotoImage(resized)

        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        x = (canvas_width - new_width) // 2 + self.offset_x
        y = (canvas_height - new_height) // 2 + self.offset_y

        self.delete("all")
        self.image_id = self.create_image(x, y, anchor='nw', image=self.displayed_image)

    def start_pan(self, event):
        self.pan_start_x = event.x - self.offset_x
        self.pan_start_y = event.y - self.offset_y

    def pan(self, event):
        self.offset_x = event.x - self.pan_start_x
        self.offset_y = event.y - self.pan_start_y
        self.update_view()

    def zoom(self, event):
        if not self.original_image:
            return

        if event.num == 5 or event.delta < 0:  # Zoom out
            self.zoom_factor *= 0.9
        else:  # Zoom in
            self.zoom_factor *= 1.1

        self.zoom_factor = max(0.01, min(30.0, self.zoom_factor))
        self.update_view()

    def on_resize(self, event):
        self.fit_to_window()

# -------------------- Palette Preview --------------------

class PalettePreview(ctk.CTkFrame):
    """Widget to display a color palette"""
    def __init__(self, master, palette, width=200, height=30, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.palette = palette
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.after(100, self.draw_palette)
        self.bind("<Configure>", lambda event: self.after(100, self.draw_palette))

    def draw_palette(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        num_colors = len(self.palette)
        if num_colors == 0:
            return
        segment_width = width / num_colors

        for i, color in enumerate(self.palette):
            x1 = i * segment_width
            x2 = (i + 1) * segment_width
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            self.canvas.create_rectangle(x1, 0, x2, height, fill=hex_color, outline='')

# -------------------- Palette Dialog --------------------

class PaletteDialog(ctk.CTkToplevel):
    def __init__(self, parent, image, custom_palettes: List[Tuple[str, List[Tuple[int, int, int]]]], save_callback):
        super().__init__(parent)
        self.title("Select Palette")
        self.geometry("600x600")
        self.image = image
        self.selected_palette = None
        self.selected_palette_name = None
        self.custom_palettes = custom_palettes
        self.save_callback = save_callback

        # Make dialog modal and stay on top
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        # Generate palette options
        self.palettes = self.generate_palettes()

        # Scroll frame
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add palette options
        self.create_palette_options()

        # Frame for the three buttons (horizontal layout)
        self.custom_buttons_frame = ctk.CTkFrame(self)
        self.custom_buttons_frame.pack(pady=10, fill="x")

        # Create Custom Palette button
        self.create_custom_palette_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Create Custom Palette",
            command=self.create_custom_palette
        )
        self.create_custom_palette_button.pack(side="left", padx=5, fill="x", expand=True)

        # Import from lospec.com button
        self.import_palette_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Import from lospec.com",
            command=self.import_from_lospec
        )
        self.import_palette_button.pack(side="left", padx=5, fill="x", expand=True)

        # Create from Image button
        self.create_from_image_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Create from Image",
            command=self.create_palette_from_image
        )
        self.create_from_image_button.pack(side="left", padx=5, fill="x", expand=True)

        # Buttons (OK/Cancel)
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(fill="x", padx=10, pady=5)

        self.cancel_button = ctk.CTkButton(
            self.button_frame, text="Cancel", command=self.cancel
        )
        self.cancel_button.pack(side="left", padx=5)

        self.ok_button = ctk.CTkButton(
            self.button_frame, text="OK", command=self.confirm
        )
        self.ok_button.pack(side="right", padx=5)

    def generate_palettes(self) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
        palettes = []
        try:
            num_colors = int(self.master.colors_entry.get())
        except ValueError:
            num_colors = 16

        # Algorithmic palettes
        median_cut = ColorReducer.reduce_colors(self.image, num_colors)
        palettes.append(("Median Cut", median_cut))

        kmeans1 = self.generate_kmeans_palette(self.image, num_colors, random_state=42)
        kmeans2 = self.generate_kmeans_palette(self.image, num_colors, random_state=123)
        palettes.append(("K-means (Variant 1)", kmeans1))
        palettes.append(("K-means (Variant 2)", kmeans2))

        uniform_palette = self.generate_uniform_palette(num_colors)
        palettes.append(("Uniform", uniform_palette))
        return palettes

    def generate_kmeans_palette(self, image: Image.Image, num_colors, random_state=42) -> List[Tuple[int, int, int]]:
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        kmeans = KMeans(n_clusters=num_colors, random_state=random_state)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in centers]

    def generate_uniform_palette(self, num_colors) -> List[Tuple[int, int, int]]:
        colors = []
        cube_size = int(math.ceil(num_colors ** (1/3)))
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    if len(colors) >= num_colors:
                        break
                    colors.append((
                        int(r * 255 / (cube_size - 1)) if cube_size > 1 else 128,
                        int(g * 255 / (cube_size - 1)) if cube_size > 1 else 128,
                        int(b * 255 / (cube_size - 1)) if cube_size > 1 else 128
                    ))
        return colors[:num_colors]

    def create_palette_options(self):
        self.selected_var = tk.StringVar(value="Median Cut")
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for name, palette in self.palettes:
            option_frame = ctk.CTkFrame(self.scroll_frame)
            option_frame.pack(fill="x", padx=5, pady=5)
            radio = ctk.CTkRadioButton(
                option_frame, text=name, variable=self.selected_var, value=name
            )
            radio.pack(side="left", padx=10)
            preview = PalettePreview(option_frame, palette)
            preview.pack(side="right", padx=10, fill="x", expand=True)

        # Custom palettes
        for name, palette in self.custom_palettes:
            option_frame = ctk.CTkFrame(self.scroll_frame)
            option_frame.pack(fill="x", padx=5, pady=5)
            radio = ctk.CTkRadioButton(
                option_frame, text=name, variable=self.selected_var, value=name
            )
            radio.pack(side="left", padx=10)
            preview = PalettePreview(option_frame, palette)
            preview.pack(side="right", padx=10, fill="x", expand=True)

    def create_custom_palette(self):
        CustomPaletteCreator(self, self.custom_palettes, self.save_callback, self.refresh_palettes)

    def create_palette_from_image(self):
        """
        Prompt user to open another image and generate a K-means(Variant 1) palette
        from it. Then show a preview dialog to confirm or pick another image again.
        """
        file_path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return  # user canceled

        # Load the image
        try:
            new_image = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}", parent=self)
            return

        # Get number of colors from parent's entry
        try:
            desired_colors = int(self.master.colors_entry.get())
            if desired_colors <= 0:
                raise ValueError
        except ValueError:
            desired_colors = 16

        # Count how many unique colors are in this image
        arr_full = np.array(new_image.convert('RGB'))
        all_pixels = arr_full.reshape(-1, 3)
        unique_pixels = np.unique(all_pixels, axis=0)
        unique_count = unique_pixels.shape[0]

        # If unique colors < desired, reduce
        if unique_count < desired_colors:
            num_colors = unique_count
        else:
            num_colors = desired_colors

        # We can't have < 1 cluster
        if num_colors < 1:
            num_colors = 1

        # Now, for K-means we might still do a random sample to keep it fast
        if len(all_pixels) > 10000:
            indices = np.random.choice(len(all_pixels), 10000, replace=False)
            pixels = all_pixels[indices]
        else:
            pixels = all_pixels

        # Perform K-means
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        kmeans_palette = [tuple(c) for c in centers]

        # Show a small preview dialog: "Use This Palette" or "Choose Another Image"
        preview_dialog = PaletteImagePreviewDialog(self, kmeans_palette, file_path, used_clusters=num_colors)
        self.wait_window(preview_dialog)

        # Check what user decided
        if preview_dialog.choose_another:
            # user wants to pick another image, so just re-run
            self.create_palette_from_image()
            return
        elif preview_dialog.use_result:
            # Insert a name for reference. 
            # We won't do an extra name prompt. We'll auto-name it.
            basename = os.path.basename(file_path)
            # palette_name = f"KMeans from {basename}"
            palette_name = f"From imported image"

            # Remove any old palette with the same name to avoid duplication
            self.palettes = [(n, p) for (n, p) in self.palettes if n != palette_name]

            # Insert the new one at the front
            self.palettes.insert(0, (palette_name, kmeans_palette))

            # Re-create the palette radio buttons so we can see the newly added palette
            self.create_palette_options()
            # Select it immediately
            self.selected_var.set(palette_name)

    def refresh_palettes(self):
        self.palettes = self.generate_palettes()
        self.create_palette_options()

    def import_from_lospec(self):
        url = simpledialog.askstring("Import Palette", "Paste lospec.com Palette URL:", parent=self)
        if not url:
            return

        try:
            parts = url.rstrip('/').split('/')
            if len(parts) < 2:
                raise ValueError("URL does not contain enough parts to extract palette name.")
            palette_slug = parts[-1]
            json_url = f'https://lospec.com/palette-list/{palette_slug}.json'
        except Exception as e:
            messagebox.showerror("Invalid URL", f"Failed to parse palette name:\n{e}", parent=self)
            return

        try:
            with urllib.request.urlopen(json_url) as response:
                data = response.read()
                palette_json = json.loads(data)
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download or parse palette JSON:\n{e}", parent=self)
            return

        try:
            name = palette_json['name']
            colors = palette_json['colors']
            rgb_colors = [self.hex_to_rgb(f"#{c}") for c in colors]
        except KeyError as e:
            messagebox.showerror("JSON Error", f"Missing key in palette JSON: {e}", parent=self)
            return
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse palette JSON:\n{e}", parent=self)
            return

        existing_names = [n for n, _ in self.palettes] + [n for n, _ in self.custom_palettes]
        if name in existing_names:
            messagebox.showerror("Duplicate Palette", f"A palette named '{name}' already exists.", parent=self)
            return

        self.custom_palettes.append((name, rgb_colors))
        self.save_callback()
        self.create_palette_options()
        self.selected_var.set(name)
        messagebox.showinfo("Success", f"Palette '{name}' imported successfully.", parent=self)

    def get_selected_palette(self) -> Optional[List[Tuple[int, int, int]]]:
        selected_name = self.selected_var.get()
        for name, palette in self.palettes:
            if name == selected_name:
                self.selected_palette_name = name
                return palette
        for name, palette in self.custom_palettes:
            if name == selected_name:
                self.selected_palette_name = name
                return palette
        return None

    def cancel(self):
        self.selected_palette = None
        self.selected_palette_name = None
        self.destroy()

    def confirm(self):
        self.selected_palette = self.get_selected_palette()
        self.destroy()

    def hex_to_rgb(self, hex_code: str) -> Tuple[int, int, int]:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# -------------------- New PaletteImagePreviewDialog --------------------

class PaletteImagePreviewDialog(ctk.CTkToplevel):
    """
    A small Toplevel to show the newly generated palette from an image
    and ask the user if they want to use it or pick another image.
    """
    def __init__(self, parent, palette: List[Tuple[int,int,int]], file_path: str, used_clusters: int):
        super().__init__(parent)
        self.title("New Palette Preview")
        self.geometry("400x180")
        self.resizable(False, False)

        # Flags for the user's choice
        self.use_result = False
        self.choose_another = False

        # Modal
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        basename = os.path.basename(file_path)
        info_label = ctk.CTkLabel(
            self,
            text=f"Generated a {used_clusters}-color palette from:\n{basename}\n\nUse this palette or pick another image?"
        )
        info_label.pack(pady=(10,0))

        self.preview = PalettePreview(self, palette, width=300, height=30)
        self.preview.pack(pady=10)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=5, fill="x")

        use_button = ctk.CTkButton(button_frame, text="Use This Palette", command=self.use_palette)
        use_button.pack(side="left", expand=True, fill="x", padx=5, pady=5)

        another_button = ctk.CTkButton(button_frame, text="Choose Another Image", command=self.pick_another)
        another_button.pack(side="right", expand=True, fill="x", padx=5, pady=5)

    def use_palette(self):
        self.use_result = True
        self.destroy()

    def pick_another(self):
        self.choose_another = True
        self.destroy()

# -------------------- New HSV Color Picker Dialog --------------------

import colorsys

PLANE_SIZE = 256  # For saturation/value plane

class HSVColorPickerDialog(ctk.CTkToplevel):
    """
    A modal dialog that provides an HSV color picker.
    The user can press "OK" to confirm the selection,
    and the chosen color is stored in self.selected_color as (R, G, B).
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("HSV Color Picker")
        self.geometry("640x400")
        self.resizable(False, False)

        # Make this dialog modal
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        # Current HSV
        self.hue = 0.0
        self.sat = 0.0
        self.val = 1.0

        # Will store the chosen RGB color as a tuple or None
        self.selected_color: Optional[Tuple[int, int, int]] = None

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Hue gradient
        self.hue_gradient_image = self.create_hue_gradient(width=360, height=20)
        self.hue_gradient_photo = ImageTk.PhotoImage(self.hue_gradient_image)
        self.hue_gradient_label = tk.Label(main_frame, image=self.hue_gradient_photo, bd=1, relief="ridge")
        self.hue_gradient_label.grid(row=0, column=0, columnspan=2, pady=(0,5), sticky="w")

        # Hue slider
        self.hue_slider = ctk.CTkSlider(
            main_frame,
            from_=0,
            to=360,
            command=self.on_hue_changed,
            width=360
        )
        self.hue_slider.set(0)
        self.hue_slider.grid(row=1, column=0, columnspan=2, padx=0, pady=(0,10), sticky="w")

        # Saturation/Value plane
        self.plane_canvas = tk.Canvas(main_frame, width=PLANE_SIZE, height=PLANE_SIZE, bd=2, relief="sunken", cursor="cross")
        self.plane_canvas.grid(row=2, column=0, padx=(0,10), pady=5)
        self.plane_canvas.bind("<Button-1>", self.on_plane_click)
        self.plane_canvas.bind("<B1-Motion>", self.on_plane_click)
        self.plane_image = None
        self.plane_photo = None
        self.circle_id = None

        # Right side color entries & preview
        self.create_color_representations(main_frame)

        # OK button to confirm
        ok_button = ctk.CTkButton(self, text="OK", command=self.on_ok)
        ok_button.pack(side="bottom", pady=(0, 10))

        # Generate plane & preview
        self.update_color_plane()
        self.update_preview()

    def create_hue_gradient(self, width=360, height=20):
        img = Image.new("RGB", (width, height), "black")
        for x in range(width):
            hue_normalized = x / float(width)  # [0..1]
            h = hue_normalized * 360
            r, g, b = colorsys.hsv_to_rgb(h/360.0, 1.0, 1.0)
            for y in range(height):
                img.putpixel((x, y), (int(r*255), int(g*255), int(b*255)))
        return img

    def on_hue_changed(self, new_hue):
        self.hue = float(new_hue)
        self.update_color_plane()
        self.update_preview()
        self.update_color_reps()

    def on_plane_click(self, event):
        x = event.x
        y = event.y
        if x < 0: x = 0
        if x >= PLANE_SIZE: x = PLANE_SIZE - 1
        if y < 0: y = 0
        if y >= PLANE_SIZE: y = PLANE_SIZE - 1

        self.sat = x / float(PLANE_SIZE - 1)
        self.val = 1.0 - (y / float(PLANE_SIZE - 1))
        self.update_preview()
        self.update_circle()
        self.update_color_reps()

    def update_color_plane(self):
        img = Image.new("RGB", (PLANE_SIZE, PLANE_SIZE), "black")
        hue_norm = self.hue / 360.0
        for j in range(PLANE_SIZE):
            v = 1.0 - j / float(PLANE_SIZE - 1)
            for i in range(PLANE_SIZE):
                s = i / float(PLANE_SIZE - 1)
                r, g, b = colorsys.hsv_to_rgb(hue_norm, s, v)
                img.putpixel((i, j), (int(r*255), int(g*255), int(b*255)))

        self.plane_image = img
        self.plane_photo = ImageTk.PhotoImage(img)
        self.plane_canvas.create_image(0, 0, anchor="nw", image=self.plane_photo)
        self.update_circle()

    def update_preview(self):
        r, g, b = self.get_rgb()
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        if hasattr(self, 'preview_box'):
            self.preview_box.configure(fg_color=hex_color)

    def update_circle(self):
        if self.plane_photo is None:
            return
        if self.circle_id is not None:
            self.plane_canvas.delete(self.circle_id)
            self.circle_id = None

        x = self.sat * (PLANE_SIZE - 1)
        y = (1.0 - self.val) * (PLANE_SIZE - 1)
        radius = 5
        x0 = x - radius
        y0 = y - radius
        x1 = x + radius
        y1 = y + radius

        try:
            bg_color = self.plane_image.getpixel((int(x), int(y)))
            luminance = 0.2126*bg_color[0] + 0.7152*bg_color[1] + 0.0722*bg_color[2]
            if luminance > 128:
                outline_color = "#000000"
            else:
                outline_color = "#FFFFFF"
        except:
            outline_color = "#FFFFFF"

        self.circle_id = self.plane_canvas.create_oval(
            x0, y0, x1, y1,
            outline=outline_color,
            width=2
        )

    def create_color_representations(self, parent):
        rep_frame = ctk.CTkFrame(parent)
        rep_frame.grid(row=2, column=1, padx=10, pady=5, sticky="n")

        # RGB Label
        rgb_label = ctk.CTkLabel(rep_frame, text="RGB:")
        rgb_label.grid(row=0, column=0, padx=5, pady=(0,5), sticky="w")

        self.r_var = tk.StringVar(value="255")
        self.g_var = tk.StringVar(value="255")
        self.b_var = tk.StringVar(value="255")

        self.r_entry = ctk.CTkEntry(rep_frame, textvariable=self.r_var, width=60)
        self.g_entry = ctk.CTkEntry(rep_frame, textvariable=self.g_var, width=60)
        self.b_entry = ctk.CTkEntry(rep_frame, textvariable=self.b_var, width=60)

        self.r_entry.bind("<Return>", self.on_rgb_enter)
        self.g_entry.bind("<Return>", self.on_rgb_enter)
        self.b_entry.bind("<Return>", self.on_rgb_enter)

        self.r_entry.grid(row=0, column=1, padx=5, pady=(0,5))
        self.g_entry.grid(row=0, column=2, padx=5, pady=(0,5))
        self.b_entry.grid(row=0, column=3, padx=5, pady=(0,5))

        # HEX Label
        hex_label = ctk.CTkLabel(rep_frame, text="HEX:")
        hex_label.grid(row=1, column=0, padx=5, pady=(10,5), sticky="w")

        self.hex_var = tk.StringVar(value="#FFFFFF")
        self.hex_entry = ctk.CTkEntry(rep_frame, textvariable=self.hex_var, width=180)
        self.hex_entry.bind("<Return>", self.on_hex_enter)
        self.hex_entry.grid(row=1, column=1, columnspan=3, padx=(5,0), pady=(10,5), sticky="w")

        # Preview
        preview_label = ctk.CTkLabel(rep_frame, text="Selected Color:")
        preview_label.grid(row=2, column=0, padx=5, pady=(10,5), sticky="w")

        self.preview_box = ctk.CTkLabel(rep_frame, text="", width=80, height=40, fg_color="#ffffff", corner_radius=6)
        self.preview_box.grid(row=2, column=1, padx=5, pady=(10,5), sticky="w")

    def get_rgb(self) -> Tuple[int, int, int]:
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(self.hue/360.0, self.sat, self.val)
        return (int(r*255), int(g*255), int(b*255))

    def update_color_reps(self):
        r, g, b = self.get_rgb()
        self.r_var.set(str(r))
        self.g_var.set(str(g))
        self.b_var.set(str(b))
        self.hex_var.set("#{:02x}{:02x}{:02x}".format(r, g, b))

    def on_rgb_enter(self, event):
        try:
            r = int(self.r_var.get())
            g = int(self.g_var.get())
            b = int(self.b_var.get())
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB must be in [0..255].")

            # Convert to HSV
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            self.hue = h * 360
            self.sat = s
            self.val = v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as ex:
            messagebox.showerror("Invalid Input", str(ex))

    def on_hex_enter(self, event):
        try:
            hex_code = self.hex_var.get().strip()
            if not (hex_code.startswith('#') and len(hex_code) == 7):
                raise ValueError("HEX code must be #RRGGBB.")
            r = int(hex_code[1:3], 16)
            g = int(hex_code[3:5], 16)
            b = int(hex_code[5:7], 16)
            import colorsys
            h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            self.hue = h * 360
            self.sat = s
            self.val = v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as ex:
            messagebox.showerror("Invalid Input", str(ex))

    def on_ok(self):
        # Finalize selection
        self.selected_color = self.get_rgb()
        self.destroy()

# -------------------- Custom Palette Creator --------------------

class CustomPaletteCreator(ctk.CTkToplevel):
    """A window for creating a custom palette with a user-friendly interface."""
    def __init__(self, parent, custom_palettes, save_callback, refresh_callback):
        super().__init__(parent)
        self.title("Create Custom Palette")
        self.geometry("500x400")
        self.resizable(False, False)

        # Make it modal
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.custom_palettes = custom_palettes
        self.save_callback = save_callback
        self.refresh_callback = refresh_callback

        self.colors: List[Tuple[int, int, int]] = []

        self.palette_frame = ctk.CTkFrame(self)
        self.palette_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.update_palette_display()

        self.save_button = ctk.CTkButton(
            self,
            text="Save Palette",
            command=self.save_palette
        )
        self.save_button.pack(pady=10)

    def update_palette_display(self):
        for widget in self.palette_frame.winfo_children():
            widget.destroy()

        square_size = 40
        padding = 5

        for idx, color in enumerate(self.colors):
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            btn = tk.Button(
                self.palette_frame,
                bg=hex_color,
                width=4,
                height=2,
                relief='raised',
                cursor="hand2"
            )
            btn.grid(row=idx // 10, column=idx % 10, padx=padding, pady=padding)
            btn.bind("<Button-3>", lambda event, index=idx: self.delete_color(index))

        plus_btn = ctk.CTkButton(
            self.palette_frame,
            text="+",
            width=square_size,
            height=square_size,
            command=self.add_color,
            corner_radius=8
        )
        plus_btn.grid(row=(len(self.colors) // 10), column=(len(self.colors) % 10), padx=padding, pady=padding)

    def add_color(self):
        """Open the custom HSV color picker dialog, then add the picked color if confirmed."""
        picker = HSVColorPickerDialog(self)
        picker.wait_window()  # Wait until the user closes the picker
        if picker.selected_color is not None:
            # The user clicked OK
            self.colors.append(picker.selected_color)
            self.update_palette_display()

    def delete_color(self, index: int):
        if 0 <= index < len(self.colors):
            del self.colors[index]
            self.update_palette_display()

    def save_palette(self):
        if not self.colors:
            messagebox.showwarning("No Colors", "Please add at least one color to the palette.", parent=self)
            return

        palette_name = simpledialog.askstring("Palette Name", "Enter a name for the custom palette:", parent=self)
        if not palette_name:
            return

        existing_names = [name for name, _ in self.custom_palettes]
        if palette_name in existing_names:
            messagebox.showerror("Duplicate Name", "A palette with this name already exists.", parent=self)
            return

        self.custom_palettes.append((palette_name, self.colors.copy()))
        self.save_callback()
        self.refresh_callback()
        self.destroy()

# -------------------- Main Application --------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Dithering Tool")
        self.geometry("1200x800")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar.grid_rowconfigure(14, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.image_viewer = ZoomableImage(self.main_frame, bg="gray20", highlightthickness=0)
        self.image_viewer.grid(row=0, column=0, sticky="nsew")

        self.create_controls()

        self.current_image = None
        self.original_filename = None
        self.dithered_image = None
        self.pixelized_image = None
        self.display_state = "pixelized"
        self.current_palette_name: Optional[str] = None

        self.palette_file = "palette.json"
        self.custom_palettes: List[Tuple[str, List[Tuple[int, int, int]]]] = []
        self.load_custom_palettes()

    def create_controls(self):
        row = 0

        self.open_button = ctk.CTkButton(
            self.sidebar, text="Open Image", command=self.open_image
        )
        self.open_button.grid(row=row, column=0, padx=20, pady=(10, 5), sticky="ew")
        row += 1

        self.mode_label = ctk.CTkLabel(self.sidebar, text="Dither Mode:")
        self.mode_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1

        self.dither_mode = ctk.CTkOptionMenu(
            self.sidebar,
            values=[mode.value for mode in DitherMode]
        )
        self.dither_mode.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.dither_mode.set(DitherMode.BAYER4x4.value)
        row += 1

        self.colors_label = ctk.CTkLabel(self.sidebar, text="Number of Colors:")
        self.colors_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1

        self.colors_entry = ctk.CTkEntry(self.sidebar, placeholder_text="16")
        self.colors_entry.insert(0, "16")
        self.colors_entry.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        self.auto_pixelize_var = tk.BooleanVar(value=True)
        self.auto_pixelize_check = ctk.CTkCheckBox(
            self.sidebar, text="Automatic Pixelization", variable=self.auto_pixelize_var,
            command=self.toggle_auto_pixelization
        )
        self.auto_pixelize_check.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="w")
        row += 1

        self.max_size_label = ctk.CTkLabel(self.sidebar, text="Maximum Size:")
        self.max_size_entry = ctk.CTkEntry(self.sidebar, placeholder_text="640")
        self.max_size_entry.insert(0, "640")

        self.max_size_label.grid(row=row, column=0, padx=20, pady=(0,0), sticky="w")
        self.max_size_entry.grid(row=row+1, column=0, padx=20, pady=(0,10), sticky="ew")
        self.max_size_label.grid_remove()
        self.max_size_entry.grid_remove()
        row += 2

        self.apply_button = ctk.CTkButton(
            self.sidebar, text="Apply Dithering", command=self.show_palette_dialog
        )
        self.apply_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        self.save_button = ctk.CTkButton(
            self.sidebar, text="Save Image", command=self.save_image
        )
        self.save_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        self.pixelize_button = ctk.CTkButton(
            self.sidebar, text="Pixelize", command=self.pixelize_image
        )
        self.pixelize_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        self.reset_button = ctk.CTkButton(
            self.sidebar, text="Fit to Window", command=self.fit_to_window
        )
        self.reset_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        self.toggle_button = ctk.CTkButton(
            self.sidebar, text="Toggle View", command=self.toggle_view
        )
        self.toggle_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        for i in range(row, 15):
            self.sidebar.grid_rowconfigure(i, weight=1)

    def toggle_auto_pixelization(self):
        if self.auto_pixelize_var.get():
            self.max_size_label.grid_remove()
            self.max_size_entry.grid_remove()
        else:
            self.max_size_label.grid()
            self.max_size_entry.grid()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.original_filename = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image:\n{e}")
                return

            if self.auto_pixelize_var.get():
                self.pixelize_image(auto=True)
            else:
                self.pixelized_image = self.current_image.convert("RGB")
                self.display_state = "pixelized"
                self.image_viewer.set_image(self.pixelized_image)
                self.fit_to_window()
                self.dithered_image = None

    def show_palette_dialog(self):
        if not self.pixelized_image:
            messagebox.showwarning("No Pixelized Image", "Please pixelize the image first.")
            return

        dialog = PaletteDialog(self, self.pixelized_image, self.custom_palettes, self.save_custom_palettes)
        self.wait_window(dialog)

        if dialog.selected_palette:
            try:
                num_colors = int(self.colors_entry.get())
                if num_colors <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Number of Colors", "Please enter a valid positive integer.")
                return

            try:
                dither_mode = DitherMode(self.dither_mode.get())
            except ValueError:
                messagebox.showerror("Invalid Dither Mode", "Please select a valid dither mode.")
                return

            self.current_palette_name = dialog.selected_palette_name or "UnknownPalette"
            ditherer = ImageDitherer(num_colors=num_colors, dither_mode=dither_mode, palette=dialog.selected_palette)
            try:
                self.dithered_image = ditherer.apply_dithering(self.pixelized_image)
            except Exception as e:
                messagebox.showerror("Dithering Error", f"An error occurred:\n{e}")
                return
            self.display_state = "dithered"
            self.image_viewer.set_image(self.dithered_image)
            self.fit_to_window()

    def pixelize_image(self, auto=False):
        if not self.current_image:
            if not auto:
                messagebox.showwarning("No Image", "Please open an image first.")
            return

        if self.auto_pixelize_var.get() or auto:
            max_size = 640
        else:
            try:
                max_size = int(self.max_size_entry.get())
                if max_size <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Maximum Size", "Please enter a valid positive integer.")
                return

        original_aspect_ratio = self.current_image.width / self.current_image.height
        if self.current_image.width >= self.current_image.height:
            new_width = max_size
            new_height = int(max_size / original_aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * original_aspect_ratio)

        resized_image = self.current_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        final_image = resized_image.convert("RGB")
        self.pixelized_image = final_image
        self.display_state = "pixelized"
        self.image_viewer.set_image(self.pixelized_image)
        self.fit_to_window()
        self.dithered_image = None

        if not auto:
            messagebox.showinfo("Pixelization Complete", "Image has been pixelized.")

    def save_image(self):
        if self.display_state == "dithered":
            image_to_save = self.dithered_image
        elif self.display_state == "pixelized":
            image_to_save = self.pixelized_image
        else:
            image_to_save = None

        if not image_to_save:
            messagebox.showwarning("No Image to Save", "There is no image to save.")
            return

        if self.original_filename:
            base_name = os.path.splitext(os.path.basename(self.original_filename))[0]
        else:
            base_name = "untitled"

        parts = [base_name]
        if self.display_state == "pixelized":
            parts.append("pixelized")
        elif self.display_state == "dithered":
            parts.append("dithered")
            parts.append(self.dither_mode.get())

        if self.current_palette_name:
            safe_palette_name = self.current_palette_name.replace(' ', '_')
            parts.append(safe_palette_name)

        try:
            num_colors = int(self.colors_entry.get())
        except ValueError:
            num_colors = 16
        parts.append(f"{num_colors}colors")

        default_filename = '_'.join(parts) + ".png"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                image_to_save.save(file_path)
                messagebox.showinfo("Image Saved", f"Image saved as: {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def fit_to_window(self):
        self.image_viewer.fit_to_window()

    def toggle_view(self):
        if self.display_state == "pixelized":
            if self.dithered_image:
                self.image_viewer.set_image(self.dithered_image)
                self.display_state = "dithered"
            else:
                messagebox.showwarning("No Dithered Image", "Please apply dithering first.")
        elif self.display_state == "dithered":
            self.image_viewer.set_image(self.pixelized_image)
            self.display_state = "pixelized"
        self.fit_to_window()

    def load_custom_palettes(self):
        if not os.path.exists(self.palette_file):
            with open(self.palette_file, 'w') as f:
                json.dump([], f)
            return
        try:
            with open(self.palette_file, 'r') as f:
                data = json.load(f)
            for palette in data:
                name = palette['name']
                colors = [self.hex_to_rgb(c) for c in palette['colors']]
                self.custom_palettes.append((name, colors))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load custom palettes:\n{e}")
            self.custom_palettes = []

    def save_custom_palettes(self):
        data = []
        for name, colors in self.custom_palettes:
            hex_colors = [self.rgb_to_hex(c) for c in colors]
            data.append({"name": name, "colors": hex_colors})
        try:
            with open(self.palette_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save custom palettes:\n{e}")

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def hex_to_rgb(self, hex_code: str) -> Tuple[int, int, int]:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# -------------------- CLI Tool --------------------

def run_cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Image Dithering Tool CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Pixelize Subcommand
    pixelize_parser = subparsers.add_parser(
        'pixelize',
        help='Pixelize an image by resizing it using nearest neighbor interpolation.'
    )
    pixelize_parser.add_argument('input_image', type=str, help='Path to the input image.')
    pixelize_parser.add_argument('output_image', type=str, help='Path to save the pixelized image.')
    pixelize_parser.add_argument('-m', '--max-size', type=int, default=640,
                                 help='Max size for pixelization (width or height).')

    # Dither Subcommand
    dither_parser = subparsers.add_parser(
        'dither',
        help='Apply dithering to an image using a specified dithering mode and palette.'
    )
    dither_parser.add_argument('input_image', type=str, help='Path to the input image.')
    dither_parser.add_argument('output_image', type=str, help='Path to save the dithered image.')
    dither_parser.add_argument('-d', '--mode', type=str, choices=[mode.value for mode in DitherMode],
                               default="bayer4x4", help='Dithering mode to use.')
    dither_parser.add_argument('-c', '--colors', type=int, default=16,
                               help='Number of colors in the palette.')
    dither_parser.add_argument('--algo-palette', type=str,
                               choices=["median_cut", "kmeans_variant1", "kmeans_variant2", "uniform"],
                               default=None, help="Algorithmic palette: median_cut, kmeans_variant1, etc.")
    dither_parser.add_argument('-p', '--palette', type=str, default=None,
                               help='Name of the custom palette to use (from palette.json).')

    # Dither-Pixelize Subcommand
    dither_pixelize_parser = subparsers.add_parser(
        'dither-pixelize',
        help='Pixelize an image then apply dithering.'
    )
    dither_pixelize_parser.add_argument('input_image', type=str, help='Path to the input image.')
    dither_pixelize_parser.add_argument('output_image', type=str, help='Path to save the processed image.')
    dither_pixelize_parser.add_argument('-d', '--mode', type=str,
                                        choices=[mode.value for mode in DitherMode],
                                        default="bayer4x4", help='Dithering mode.')
    dither_pixelize_parser.add_argument('-c', '--colors', type=int, default=16,
                                        help='Number of colors in the palette.')
    dither_pixelize_parser.add_argument('--algo-palette', type=str,
                                        choices=["median_cut", "kmeans_variant1", "kmeans_variant2", "uniform"],
                                        default=None, help="Algorithmic palette.")
    dither_pixelize_parser.add_argument('-p', '--palette', type=str, default=None,
                                        help='Name of the custom palette to use (from palette.json).')
    dither_pixelize_parser.add_argument('-m', '--max-size', type=int, default=640,
                                        help='Max size for pixelization (width or height).')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # CLI Helper
    def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        hex_code = hex_code.lstrip('#')
        if len(hex_code) != 6:
            raise ValueError(f"Invalid hex code: {hex_code}")
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    def load_custom_palettes(palette_file: str) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
        custom_palettes = []
        if os.path.exists(palette_file):
            try:
                with open(palette_file, 'r') as f:
                    data = json.load(f)
                for p in data:
                    name = p['name']
                    colors = [hex_to_rgb(c) for c in p['colors']]
                    custom_palettes.append((name, colors))
            except Exception as e:
                print(f"Warning: Failed to load custom palettes: {e}")
        else:
            with open(palette_file, 'w') as f:
                json.dump([], f)
        return custom_palettes

    custom_palettes = load_custom_palettes("palette.json")

    from PIL import Image

    def pixelize_image(image: Image.Image, max_size: int) -> Image.Image:
        original_aspect_ratio = image.width / image.height
        if image.width >= image.height:
            new_width = max_size
            new_height = int(max_size / original_aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * original_aspect_ratio)
        resized = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        return resized.convert("RGB")

    def generate_kmeans_palette(image: Image.Image, num_colors: int, random_state=42) -> List[Tuple[int,int,int]]:
        arr = np.array(image.convert('RGB'))
        pixels = arr.reshape(-1,3)
        if len(pixels) > 10000:
            idx = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[idx]
        kmeans = KMeans(n_clusters=num_colors, random_state=random_state)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in centers]

    def generate_uniform_palette(num_colors: int) -> List[Tuple[int,int,int]]:
        colors = []
        cube_size = int(math.ceil(num_colors ** (1/3)))
        for r in range(cube_size):
            for g in range(cube_size):
                for b in range(cube_size):
                    if len(colors) >= num_colors:
                        break
                    colors.append((
                        int(r * 255 / (cube_size - 1)) if cube_size > 1 else 128,
                        int(g * 255 / (cube_size - 1)) if cube_size > 1 else 128,
                        int(b * 255 / (cube_size - 1)) if cube_size > 1 else 128
                    ))
        return colors[:num_colors]

    def generate_median_cut_palette(image: Image.Image, num_colors: int) -> List[Tuple[int,int,int]]:
        image = image.convert('RGB')
        c = list(set(image.getdata()))
        if num_colors < 1:
            num_colors = 1
        depth = int(math.log2(num_colors)) if num_colors>1 else 0
        def find_dominant_channel(cs):
            rng = [0,0,0]
            for chan in range(3):
                mn = min(col[chan] for col in cs)
                mx = max(col[chan] for col in cs)
                rng[chan] = mx - mn
            return rng.index(max(rng))
        def median_cut(cs, d):
            if d==0 or len(cs)==0:
                if len(cs)==0:
                    return [(0,0,0)]
                avg = tuple(int(sum(ch)/len(cs)) for ch in zip(*cs))
                return [avg]
            channel = find_dominant_channel(cs)
            cs.sort(key=lambda x:x[channel])
            mid = len(cs)//2
            return median_cut(cs[:mid], d-1)+median_cut(cs[mid:], d-1)
        return median_cut(c, depth)

    def get_algorithmic_palette(img: Image.Image, algo: str, num_colors: int) -> List[Tuple[int,int,int]]:
        if algo == "median_cut":
            return generate_median_cut_palette(img, num_colors)
        elif algo == "kmeans_variant1":
            return generate_kmeans_palette(img, num_colors, 42)
        elif algo == "kmeans_variant2":
            return generate_kmeans_palette(img, num_colors, 123)
        elif algo == "uniform":
            return generate_uniform_palette(num_colors)
        else:
            return generate_median_cut_palette(img, num_colors)

    if args.command == 'pixelize':
        try:
            img = Image.open(args.input_image)
            pix = pixelize_image(img, args.max_size)
            pix.save(args.output_image)
            print(f"Pixelized image saved to {args.output_image}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == 'dither':
        try:
            img = Image.open(args.input_image)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        if args.algo_palette and args.palette:
            print("Error: Choose either --algo-palette or --palette, not both.")
            sys.exit(1)
        if args.palette:
            pal = None
            for n,c in custom_palettes:
                if n.lower()==args.palette.lower():
                    pal=c
                    break
            if pal is None:
                print(f"Error: custom palette '{args.palette}' not found.")
                sys.exit(1)
            selected_palette = pal
        elif args.algo_palette:
            selected_palette = get_algorithmic_palette(img, args.algo_palette, args.colors)
        else:
            selected_palette = generate_median_cut_palette(img, args.colors)
        try:
            dmode = DitherMode(args.mode)
        except:
            print(f"Error: Unsupported Dither Mode {args.mode}")
            sys.exit(1)
        dith = ImageDitherer(args.colors, dmode, selected_palette)
        try:
            out = dith.apply_dithering(img)
            out.save(args.output_image)
            print(f"Dithered image saved to {args.output_image}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == 'dither-pixelize':
        try:
            img = Image.open(args.input_image)
            pix = pixelize_image(img, args.max_size)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        if args.algo_palette and args.palette:
            print("Error: Choose either --algo-palette or --palette, not both.")
            sys.exit(1)
        if args.palette:
            pal = None
            for n,c in custom_palettes:
                if n.lower()==args.palette.lower():
                    pal=c
                    break
            if pal is None:
                print(f"Error: custom palette '{args.palette}' not found.")
                sys.exit(1)
            selected_palette = pal
        elif args.algo_palette:
            selected_palette = get_algorithmic_palette(pix, args.algo_palette, args.colors)
        else:
            selected_palette = generate_median_cut_palette(pix, args.colors)
        try:
            dmode = DitherMode(args.mode)
        except:
            print(f"Error: Unsupported Dither Mode {args.mode}")
            sys.exit(1)
        dith = ImageDitherer(args.colors, dmode, selected_palette)
        try:
            out = dith.apply_dithering(pix)
            out.save(args.output_image)
            print(f"Pixelized + dithered image saved to {args.output_image}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Unknown command.")
        parser.print_help()
        sys.exit(1)

def main():
    import sys
    if len(sys.argv) > 1:
        run_cli()
    else:
        app = App()
        app.mainloop()

if __name__ == "__main__":
    main()
