import json
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
from enum import Enum
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
import math
import os

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
            # Query for the nearest neighbor only (k=1)
            _, idx = tree.query(flat_pixels, k=1, workers=-1)

            # FIX: When k=1, idx is a 1D array. Do not use idx[:, 0].
            dithered_pixels = palette_arr[idx, :]

            # Reshape back to image dimensions and convert to uint8
            out_arr = dithered_pixels.reshape((h, w, 3)).astype(np.uint8)

            # Create and return the final image
            result = Image.fromarray(out_arr, mode='RGB')
            return result
        else:
            # -------------------- Handling Dither Modes with Threshold --------------------
            # Query for the 2 nearest neighbors (k=2)
            distances, indices = tree.query(flat_pixels, k=2, workers=-1)
            distances_sq = distances ** 2  # Squared distances

            dist_nearest = distances_sq[:, 0]
            dist_second = distances_sq[:, 1]

            # Compute the dithering factor = dist_nearest / (dist_nearest + dist_second)
            total_dist = dist_nearest + dist_second
            factor = np.where(total_dist == 0, 0.0, dist_nearest / total_dist)

            # Tile the threshold matrix to match image dimensions
            th_mat = self.threshold_matrix
            th_h, th_w = th_mat.shape
            tiled_threshold = np.tile(th_mat, ((h + th_h - 1) // th_h, (w + th_w - 1) // th_w))
            tiled_threshold = tiled_threshold[:h, :w]
            flat_threshold = tiled_threshold.flatten()

            # Decide whether to use the nearest or second-nearest palette color
            idx_nearest = indices[:, 0]
            idx_second = indices[:, 1]
            use_nearest = (factor <= flat_threshold)
            final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)

            # Gather the final palette colors
            dithered_pixels = palette_arr[final_indices, :]

            # Reshape back to image dimensions and convert to uint8
            out_arr = dithered_pixels.reshape((h, w, 3)).astype(np.uint8)

            # Create and return the final image
            result = Image.fromarray(out_arr, mode='RGB')
            return result

# -------------------- Image Viewer --------------------

class ZoomableImage(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        
        # Image handling
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
            
        # Ensure the canvas size is updated
        self.update_idletasks()

        # Get canvas size
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        # Get image size
        image_width, image_height = self.original_image.size
        
        # Calculate scaling factors
        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        
        # Use smaller ratio to fit image entirely
        self.zoom_factor = min(width_ratio, height_ratio)

        # Reset offset to center the image
        self.offset_x = 0
        self.offset_y = 0
        
        self.update_view()

    def update_view(self):
        """Update the displayed image based on current zoom and pan"""
        if not self.original_image:
            return
            
        # Calculate new size
        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)
        
        if new_width <= 0 or new_height <= 0:
            return
            
        # Resize image
        resized = self.original_image.resize((new_width, new_height), Image.Resampling.NEAREST)
        
        # Update PhotoImage
        self.displayed_image = ImageTk.PhotoImage(resized)
        
        # Calculate center position
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        x = (canvas_width - new_width) // 2 + self.offset_x
        y = (canvas_height - new_height) // 2 + self.offset_y
        
        # Update canvas
        self.delete("all")
        self.image_id = self.create_image(x, y, anchor='nw', image=self.displayed_image)

    def start_pan(self, event):
        """Record starting position for pan operation"""
        self.pan_start_x = event.x - self.offset_x
        self.pan_start_y = event.y - self.offset_y

    def pan(self, event):
        """Update image position during pan"""
        self.offset_x = event.x - self.pan_start_x
        self.offset_y = event.y - self.pan_start_y
        self.update_view()

    def zoom(self, event):
        """Handle zoom events from mouse wheel"""
        if not self.original_image:
            return
            
        # Calculate zoom factor
        if event.num == 5 or event.delta < 0:  # Zoom out
            self.zoom_factor *= 0.9
        else:  # Zoom in
            self.zoom_factor *= 1.1
            
        # Limit zoom range
        self.zoom_factor = max(0.01, min(30.0, self.zoom_factor))
        
        self.update_view()

    def on_resize(self, event):
        """Handle window resize events"""
        self.fit_to_window()



# -------------------- Palette Preview --------------------

class PalettePreview(ctk.CTkFrame):
    """Widget to display a color palette"""
    def __init__(self, master, palette, width=200, height=30, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.palette = palette
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_palette()
        self.bind("<Configure>", lambda event: self.draw_palette())  # Redraw on resize

    def draw_palette(self):
        """Draw the palette colors in equal segments"""
        self.canvas.delete("all")  # Clear previous drawings
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
        self.geometry("600x600")  # Increased height to accommodate new widgets
        
        self.image = image
        self.selected_palette = None
        self.selected_palette_name = None  # We'll store the "label" (name) here.
        self.custom_palettes = custom_palettes  # Reference to App's custom palettes
        self.save_callback = save_callback      # Callback to save palettes
        
        # Generate different palette options
        self.palettes = self.generate_palettes()
        
        # Create scrollable frame for palette options
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add palette options
        self.create_palette_options()
        
        # Add 'Create Custom Palette' button
        self.create_custom_palette_button = ctk.CTkButton(
            self,
            text="Create Custom Palette",
            command=self.create_custom_palette
        )
        self.create_custom_palette_button.pack(pady=10)
        
        # Add buttons
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
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()

    # @staticmethod
    def generate_palettes(self) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
        """Generate different palette options"""
        palettes = []
        try:
            num_colors = int(self.master.colors_entry.get())
        except ValueError:
            num_colors = 16  # Fallback to default if invalid
        
        # ---- Algorithmic Methods ----
        
        # Median Cut (original)
        median_cut_palette = ColorReducer.reduce_colors(self.image, num_colors)
        palettes.append(("Median Cut", median_cut_palette))
        
        # K-means with different random states for variety
        kmeans_palette1 = self.generate_kmeans_palette(num_colors, random_state=42)
        kmeans_palette2 = self.generate_kmeans_palette(num_colors, random_state=123)
        palettes.append(("K-means (Variant 1)", kmeans_palette1))
        palettes.append(("K-means (Variant 2)", kmeans_palette2))
        
        # Uniform color quantization
        uniform_palette = self.generate_uniform_palette(num_colors)
        palettes.append(("Uniform", uniform_palette))
        
        return palettes

    def generate_kmeans_palette(self, num_colors, random_state=42) -> List[Tuple[int, int, int]]:
        """Generate a palette using k-means clustering"""
        # Ensure the image is in RGB mode
        img_array = np.array(self.image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)
        
        # Randomly sample pixels if image is too large
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        kmeans = KMeans(n_clusters=num_colors, random_state=random_state)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]

    def generate_uniform_palette(self, num_colors) -> List[Tuple[int, int, int]]:
        """Generate a uniform distribution of colors"""
        colors = []
        cube_size = int(math.ceil(num_colors ** (1/3)))  # Cube root for RGB distribution
        
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
        """Create radio buttons with palette previews"""
        self.selected_var = tk.StringVar(value="Median Cut")
        
        # Clear existing widgets in scroll_frame
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        # Iterate over all predefined palettes
        for name, palette in self.palettes:
            # Create frame for this option
            option_frame = ctk.CTkFrame(self.scroll_frame)
            option_frame.pack(fill="x", padx=5, pady=5)
            
            # Add radio button
            radio = ctk.CTkRadioButton(
                option_frame, 
                text=name,
                variable=self.selected_var,
                value=name
            )
            radio.pack(side="left", padx=10)
            
            # Add palette preview
            preview = PalettePreview(option_frame, palette)
            preview.pack(side="right", padx=10, fill="x", expand=True)
        
        # Iterate over custom palettes
        for custom_palette in self.custom_palettes:
            name, palette = custom_palette
            option_frame = ctk.CTkFrame(self.scroll_frame)
            option_frame.pack(fill="x", padx=5, pady=5)
            
            radio = ctk.CTkRadioButton(
                option_frame,
                text=name,
                variable=self.selected_var,
                value=name
            )
            radio.pack(side="left", padx=10)
            
            preview = PalettePreview(option_frame, palette)
            preview.pack(side="right", padx=10, fill="x", expand=True)

    def create_custom_palette(self):
        """Allow user to create a custom palette by selecting colors"""
        # Prompt for palette name
        palette_name = simpledialog.askstring("Palette Name", "Enter a name for the custom palette:", parent=self)
        if not palette_name:
            return  # User cancelled or entered nothing
        
        # Check for unique palette name
        existing_names = [name for name, _ in self.palettes] + [name for name, _ in self.custom_palettes]
        if palette_name in existing_names:
            messagebox.showerror("Duplicate Name", "A palette with this name already exists. Please choose a different name.", parent=self)
            return
        
        # Initialize list to store colors
        custom_colors = []
        
        while True:
            # Ask user how they want to add the next color
            add_method = messagebox.askquestion("Add Color", 
                                                "Do you want to add a color via the Color Chooser?\n\nSelect 'Yes' for Color Chooser or 'No' to enter a Hex Code manually.",
                                                icon='question', parent=self)
            if add_method == 'yes':
                # Open color chooser dialog
                color = colorchooser.askcolor(title=f"Select Color {len(custom_colors)+1} for '{palette_name}'", parent=self)
                if color[0] is None:
                    # User cancelled color chooser
                    add_another = messagebox.askyesno("Add Another Color", "Do you want to add another color to the palette?", parent=self)
                    if not add_another:
                        break
                    else:
                        continue
                # color[0] is (R, G, B), color[1] is hex code
                rgb_color = tuple(int(c) for c in color[0])
                custom_colors.append(rgb_color)
            else:
                # User chose to enter a hex code manually
                hex_code = simpledialog.askstring("Enter Hex Code", "Enter the color hex code (e.g., #332c50):", parent=self)
                if not hex_code:
                    # User cancelled hex code entry
                    add_another = messagebox.askyesno("Add Another Color", "Do you want to add another color to the palette?", parent=self)
                    if not add_another:
                        break
                    else:
                        continue
                # Validate hex code
                if self.validate_hex_code(hex_code):
                    rgb_color = self.hex_to_rgb(hex_code)
                    custom_colors.append(rgb_color)
                else:
                    messagebox.showerror("Invalid Hex Code", "Please enter a valid hex code in the format #RRGGBB.", parent=self)
                    # Optionally, retry adding the same color
                    continue
            
            # Ask if the user wants to add another color
            add_another = messagebox.askyesno("Add Another Color", "Do you want to add another color to the palette?", parent=self)
            if not add_another:
                break
        
        if not custom_colors:
            messagebox.showwarning("No Colors Selected", "No colors were selected for the custom palette.", parent=self)
            return
        
        # Add the custom palette to the list
        self.custom_palettes.append((palette_name, custom_colors))
        
        # Update the palette options UI
        self.create_palette_options()
        
        # Select the newly created palette
        self.selected_var.set(palette_name)
        
        # Save custom palettes via callback
        self.save_callback()

    def validate_hex_code(self, hex_code: str) -> bool:
        """Validate the hex code format"""
        if isinstance(hex_code, str) and hex_code.startswith('#') and len(hex_code) == 7:
            try:
                int(hex_code[1:], 16)
                return True
            except ValueError:
                return False
        return False

    def hex_to_rgb(self, hex_code: str) -> Tuple[int, int, int]:
        """Convert hex code to RGB tuple"""
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    
    def get_selected_palette(self) -> Optional[List[Tuple[int, int, int]]]:
        """Get the currently selected palette"""
        selected_name = self.selected_var.get()
        # Search in predefined palettes
        for name, palette in self.palettes:
            if name == selected_name:
                self.selected_palette_name = name
                return palette
        # Search in custom palettes
        for name, palette in self.custom_palettes:
            if name == selected_name:
                self.selected_palette_name = name
                return palette
        return None

    def cancel(self):
        """Cancel selection"""
        self.selected_palette = None
        self.selected_palette_name = None
        self.destroy()

    def confirm(self):
        """Confirm selection"""
        self.selected_palette = self.get_selected_palette()
        # self.selected_palette_name is also assigned inside get_selected_palette()
        self.destroy()

# -------------------- Main Application --------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Image Dithering Tool")
        self.geometry("1200x800")

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar.grid_rowconfigure(14, weight=1)  # Adjusted for new widgets

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Create image viewer
        self.image_viewer = ZoomableImage(self.main_frame, bg="gray20", highlightthickness=0)
        self.image_viewer.grid(row=0, column=0, sticky="nsew")

        # Create controls
        self.create_controls()

        # Initialize state
        self.current_image = None
        self.original_filename = None  # We'll store the original file path here
        self.dithered_image = None
        self.pixelized_image = None  # For pixelized image
        self.display_state = "pixelized"  # Track current display state

        # We'll store the current palette name when we pick one
        self.current_palette_name: Optional[str] = None

        # Load custom palettes from palette.json
        self.palette_file = "palette.json"
        self.custom_palettes: List[Tuple[str, List[Tuple[int, int, int]]]] = []
        self.load_custom_palettes()

    def create_controls(self):
        """Create sidebar controls"""
        row = 0  # Initialize row counter

        # Open image button
        self.open_button = ctk.CTkButton(
            self.sidebar, text="Open Image", command=self.open_image
        )
        self.open_button.grid(row=row, column=0, padx=20, pady=(10, 5), sticky="ew")
        row += 1

        # Dither mode selection
        self.mode_label = ctk.CTkLabel(self.sidebar, text="Dither Mode:")
        self.mode_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1
        
        self.dither_mode = ctk.CTkOptionMenu(
            self.sidebar,
            values=[mode.value for mode in DitherMode]
        )
        self.dither_mode.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.dither_mode.set(DitherMode.BAYER4x4.value)  # Set default value
        row += 1

        # Number of colors input
        self.colors_label = ctk.CTkLabel(self.sidebar, text="Number of Colors:")
        self.colors_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1
        
        self.colors_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="16"
        )
        self.colors_entry.insert(0, "16")  # Default value
        self.colors_entry.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # Automatic Pixelization check button
        self.auto_pixelize_var = tk.BooleanVar(value=True)  # Default to True
        self.auto_pixelize_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Automatic Pixelization",
            variable=self.auto_pixelize_var,
            command=self.toggle_auto_pixelization
        )
        self.auto_pixelize_check.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="w")
        row += 1

        # Maximum Size entry (hidden by default)
        self.max_size_label = ctk.CTkLabel(self.sidebar, text="Maximum Size:")
        self.max_size_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="640"
        )
        self.max_size_entry.insert(0, "640")  # Default value

        # Initially hidden
        self.max_size_label.grid(row=row, column=0, padx=20, pady=(0, 0), sticky="w")
        self.max_size_entry.grid(row=row+1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.max_size_label.grid_remove()
        self.max_size_entry.grid_remove()
        row += 2

        # Apply dithering button
        self.apply_button = ctk.CTkButton(
            self.sidebar, text="Apply Dithering", command=self.show_palette_dialog
        )
        self.apply_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        # Save image button
        self.save_button = ctk.CTkButton(
            self.sidebar, text="Save Image", command=self.save_image
        )
        self.save_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        # Pixelize button
        self.pixelize_button = ctk.CTkButton(
            self.sidebar, text="Pixelize", command=self.pixelize_image
        )
        self.pixelize_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        # Reset view button
        self.reset_button = ctk.CTkButton(
            self.sidebar, text="Fit to Window", command=self.fit_to_window
        )
        self.reset_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        # Toggle pixelized/dithered view button
        self.toggle_button = ctk.CTkButton(
            self.sidebar, text="Toggle View", command=self.toggle_view
        )
        self.toggle_button.grid(row=row, column=0, padx=20, pady=10, sticky="ew")
        row += 1

        # Adjust grid row weights if necessary
        for i in range(row, 15):
            self.sidebar.grid_rowconfigure(i, weight=1)

    def toggle_auto_pixelization(self):
        """Show or hide the Maximum Size entry based on the Automatic Pixelization checkbox"""
        if self.auto_pixelize_var.get():
            self.max_size_label.grid_remove()
            self.max_size_entry.grid_remove()
        else:
            self.max_size_label.grid()
            self.max_size_entry.grid()

    def open_image(self):
        """Open an image file and automatically pixelize it if enabled"""
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
            
            # Pixelize automatically if the checkbox is checked
            if self.auto_pixelize_var.get():
                self.pixelize_image(auto=True)
            else:
                # Display the original image without pixelization
                self.pixelized_image = self.current_image.convert("RGB")
                self.display_state = "pixelized"
                self.image_viewer.set_image(self.pixelized_image)
                self.fit_to_window()
                self.dithered_image = None  # Reset dithered image

    def show_palette_dialog(self):
        """Show palette selection dialog and apply dithering to pixelized image"""
        if not self.pixelized_image:
            messagebox.showwarning("No Pixelized Image", "Please pixelize the image first.")
            return
            
        dialog = PaletteDialog(self, self.pixelized_image, self.custom_palettes, self.save_custom_palettes)
        self.wait_window(dialog)
        
        if dialog.selected_palette:
            # Create ditherer with current settings and selected palette
            try:
                num_colors = int(self.colors_entry.get())
                if num_colors <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Number of Colors", "Please enter a valid positive integer for the number of colors.")
                return

            try:
                dither_mode = DitherMode(self.dither_mode.get())
            except ValueError:
                messagebox.showerror("Invalid Dither Mode", "Please select a valid dither mode.")
                return

            # Store the name of the chosen palette
            self.current_palette_name = dialog.selected_palette_name or "UnknownPalette"

            ditherer = ImageDitherer(
                num_colors=num_colors,
                dither_mode=dither_mode,
                palette=dialog.selected_palette
            )

            # Apply dithering to pixelized_image
            try:
                self.dithered_image = ditherer.apply_dithering(self.pixelized_image)
            except Exception as e:
                messagebox.showerror("Dithering Error", f"An error occurred during dithering:\n{e}")
                return
            self.display_state = "dithered"  # Update display state
            self.image_viewer.set_image(self.dithered_image)
            self.fit_to_window()

    def pixelize_image(self, auto=False):
        """Apply pixelization to the current image"""
        if not self.current_image:
            if not auto:
                messagebox.showwarning("No Image", "Please open an image first.")
            return

        # Define the target resolution or maximum size
        if self.auto_pixelize_var.get() or auto:
            max_size = 640  # Default value
        else:
            try:
                max_size = int(self.max_size_entry.get())
                if max_size <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Maximum Size", "Please enter a valid positive integer for the maximum size.")
                return

        # Calculate aspect ratio
        original_aspect_ratio = self.current_image.width / self.current_image.height

        # Determine new size based on the maximum dimension
        if self.current_image.width >= self.current_image.height:
            new_width = max_size
            new_height = int(max_size / original_aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * original_aspect_ratio)

        # Resize the image using nearest neighbor interpolation
        resized_image = self.current_image.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Ensure the final image is in RGB mode
        final_image = resized_image.convert("RGB")

        # Store the pixelized image
        self.pixelized_image = final_image
        self.display_state = "pixelized"  # Update display state
        self.image_viewer.set_image(self.pixelized_image)
        self.fit_to_window()
        self.dithered_image = None  # Reset dithered image

        if not auto:
            messagebox.showinfo("Pixelization Complete", "Image has been pixelized.")

    def save_image(self):
        """Save the currently displayed image based on display_state with a default filename."""
        if self.display_state == "dithered":
            image_to_save = self.dithered_image
        elif self.display_state == "pixelized":
            image_to_save = self.pixelized_image
        else:
            image_to_save = None

        if not image_to_save:
            messagebox.showwarning("No Image to Save", "There is no image to save.")
            return

        # Build a default filename
        # 1) Base name: if we have self.original_filename, use that minus extension
        if self.original_filename:
            base_name = os.path.splitext(os.path.basename(self.original_filename))[0]
        else:
            base_name = "untitled"

        # 2) Show pixelized/dithered
        parts = [base_name]
        if self.display_state == "pixelized":
            parts.append("pixelized")
        elif self.display_state == "dithered":
            parts.append("dithered")
            # Also include the dithering mode
            parts.append(self.dither_mode.get())
        
        # 3) Current palette name (if any)
        if self.current_palette_name:
            # Turn spaces etc. into underscores
            safe_palette_name = self.current_palette_name.replace(' ', '_')
            parts.append(safe_palette_name)

        # 4) Number of colors
        try:
            num_colors = int(self.colors_entry.get())
        except ValueError:
            num_colors = 16
        parts.append(f"{num_colors}colors")

        default_filename = '_'.join(parts) + ".png"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                image_to_save.save(file_path)
                messagebox.showinfo("Image Saved", f"Image saved as: {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")

    def fit_to_window(self):
        """Fit image to window"""
        self.image_viewer.fit_to_window()

    def toggle_view(self):
        """Toggle between pixelized and dithered image"""
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
        """Load custom palettes from palette.json"""
        if not os.path.exists(self.palette_file):
            # Create an empty palette.json file
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
        """Save custom palettes to palette.json"""
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
        """Convert RGB tuple to hex code"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def hex_to_rgb(self, hex_code: str) -> Tuple[int, int, int]:
        """Convert hex code to RGB tuple"""
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

    # -------------------- Pixelize Subcommand --------------------
    pixelize_parser = subparsers.add_parser(
        'pixelize',
        help='Pixelize an image by resizing it using nearest neighbor interpolation.'
    )
    pixelize_parser.add_argument(
        'input_image',
        type=str,
        help='Path to the input image.'
    )
    pixelize_parser.add_argument(
        'output_image',
        type=str,
        help='Path to save the pixelized image.'
    )
    pixelize_parser.add_argument(
        '-m', '--max-size',
        type=int,
        default=640,
        help='Maximum size (width or height) for pixelization while maintaining aspect ratio.'
    )

    # -------------------- Dither Subcommand --------------------
    dither_parser = subparsers.add_parser(
        'dither',
        help='Apply dithering to an image using a specified dithering mode and palette.'
    )
    dither_parser.add_argument(
        'input_image',
        type=str,
        help='Path to the input image.'
    )
    dither_parser.add_argument(
        'output_image',
        type=str,
        help='Path to save the dithered image.'
    )
    dither_parser.add_argument(
        '-d', '--mode',
        type=str,
        choices=[mode.value for mode in DitherMode],
        default="bayer4x4",
        help='Dithering mode to use.'
    )
    dither_parser.add_argument(
        '-c', '--colors',
        type=int,
        default=16,
        help='Number of colors in the palette.'
    )
    dither_parser.add_argument(
        '--algo-palette',
        type=str,
        choices=["median_cut", "kmeans_variant1", "kmeans_variant2", "uniform"],
        default=None,
        help="Choose an algorithmic palette from: median_cut, kmeans_variant1, kmeans_variant2, uniform."
    )
    dither_parser.add_argument(
        '-p', '--palette',
        type=str,
        default=None,
        help='Name of the custom palette to use (from palette.json).'
    )

    # -------------------- Dither-Pixelize Subcommand --------------------
    dither_pixelize_parser = subparsers.add_parser(
        'dither-pixelize',
        help='Pixelize an image and then apply dithering with specified settings.'
    )
    dither_pixelize_parser.add_argument(
        'input_image',
        type=str,
        help='Path to the input image.'
    )
    dither_pixelize_parser.add_argument(
        'output_image',
        type=str,
        help='Path to save the processed image.'
    )
    dither_pixelize_parser.add_argument(
        '-d', '--mode',
        type=str,
        choices=[mode.value for mode in DitherMode],
        default="bayer4x4",
        help='Dithering mode to use after pixelization.'
    )
    dither_pixelize_parser.add_argument(
        '-c', '--colors',
        type=int,
        default=16,
        help='Number of colors in the palette.'
    )
    dither_pixelize_parser.add_argument(
        '--algo-palette',
        type=str,
        choices=["median_cut", "kmeans_variant1", "kmeans_variant2", "uniform"],
        default=None,
        help="Choose an algorithmic palette from: median_cut, kmeans_variant1, kmeans_variant2, uniform."
    )
    dither_pixelize_parser.add_argument(
        '-p', '--palette',
        type=str,
        default=None,
        help='Name of the custom palette to use (from palette.json).'
    )
    dither_pixelize_parser.add_argument(
        '-m', '--max-size',
        type=int,
        default=640,
        help='Maximum size (width or height) for pixelization while maintaining aspect ratio.'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # -------------------- Helper Function --------------------
    def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """Convert hex code to RGB tuple"""
        hex_code = hex_code.lstrip('#')
        if len(hex_code) != 6:
            raise ValueError(f"Invalid hex code: {hex_code}")
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    # -------------------- Load Custom Palettes --------------------
    def load_custom_palettes(palette_file: str) -> List[Tuple[str, List[Tuple[int, int, int]]]]:
        """Load custom palettes from palette.json"""
        custom_palettes = []
        if os.path.exists(palette_file):
            try:
                with open(palette_file, 'r') as f:
                    data = json.load(f)
                for palette in data:
                    name = palette['name']
                    colors = [hex_to_rgb(c) for c in palette['colors']]
                    custom_palettes.append((name, colors))
            except Exception as e:
                print(f"Warning: Failed to load custom palettes from '{palette_file}': {e}")
        else:
            # Create an empty palette.json file
            with open(palette_file, 'w') as f:
                json.dump([], f)
        return custom_palettes

    custom_palettes = load_custom_palettes("palette.json")

    # -------------------- Pixelize Function --------------------
    def pixelize_image(image: Image.Image, max_size: int) -> Image.Image:
        """Resize image using nearest neighbor interpolation to pixelize it."""
        original_aspect_ratio = image.width / image.height
        if image.width >= image.height:
            new_width = max_size
            new_height = int(max_size / original_aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * original_aspect_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        resized_image = resized_image.convert("RGB")
        return resized_image

    # -------------------- Algorithmic Palette Generators --------------------
    def generate_kmeans_palette(image: Image.Image, num_colors: int, random_state=42) -> List[Tuple[int,int,int]]:
        img_array = np.array(image.convert('RGB'))
        pixels = img_array.reshape(-1, 3)
        
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        kmeans = KMeans(n_clusters=num_colors, random_state=random_state)
        kmeans.fit(pixels)
        
        centers = kmeans.cluster_centers_.astype(int)
        return [tuple(center) for center in centers]

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
        colors = list(set(image.getdata()))
        if num_colors < 1:
            num_colors = 1
        depth = int(math.log2(num_colors)) if num_colors > 1 else 0
        
        def find_dominant_channel(cs: List[Tuple[int,int,int]]) -> int:
            ranges = [0, 0, 0]
            for chan in range(3):
                cmin = min(c[chan] for c in cs)
                cmax = max(c[chan] for c in cs)
                ranges[chan] = cmax - cmin
            return ranges.index(max(ranges))
        
        def median_cut(cs: List[Tuple[int,int,int]], d: int) -> List[Tuple[int,int,int]]:
            if d == 0 or len(cs) == 0:
                if len(cs) == 0:
                    return [(0,0,0)]
                avg_c = tuple(int(sum(ch)/len(cs)) for ch in zip(*cs))
                return [avg_c]
            channel = find_dominant_channel(cs)
            cs.sort(key=lambda x: x[channel])
            mid = len(cs)//2
            return median_cut(cs[:mid], d-1) + median_cut(cs[mid:], d-1)

        return median_cut(colors, depth)

    def get_algorithmic_palette(image: Image.Image, algo: str, num_colors: int) -> List[Tuple[int,int,int]]:
        if algo == "median_cut":
            return generate_median_cut_palette(image, num_colors)
        elif algo == "kmeans_variant1":
            return generate_kmeans_palette(image, num_colors, random_state=42)
        elif algo == "kmeans_variant2":
            return generate_kmeans_palette(image, num_colors, random_state=123)
        elif algo == "uniform":
            return generate_uniform_palette(num_colors)
        else:
            return generate_median_cut_palette(image, num_colors)

    # -------------------- Main Command Handling --------------------
    if args.command == 'pixelize':
        try:
            image = Image.open(args.input_image)
        except Exception as e:
            print(f"Error: Failed to open image '{args.input_image}': {e}")
            sys.exit(1)

        try:
            pixelized_image = pixelize_image(image, args.max_size)
        except Exception as e:
            print(f"Error: Failed to pixelize image: {e}")
            sys.exit(1)

        try:
            pixelized_image.save(args.output_image)
            print(f"Success: Pixelized image saved to '{args.output_image}'.")
        except Exception as e:
            print(f"Error: Failed to save image '{args.output_image}': {e}")
            sys.exit(1)

    elif args.command == 'dither':
        try:
            image = Image.open(args.input_image)
        except Exception as e:
            print(f"Error: Failed to open image '{args.input_image}': {e}")
            sys.exit(1)

        if args.algo_palette and args.palette:
            print("Error: You cannot specify both --algo-palette and --palette. Choose one.")
            sys.exit(1)

        if args.palette:
            # look for custom
            selected_palette = None
            found_custom = False
            for name, pal in custom_palettes:
                if name.lower() == args.palette.lower():
                    selected_palette = pal
                    found_custom = True
                    break
            if not found_custom:
                print(f"Error: Custom palette '{args.palette}' not found in palette.json.")
                sys.exit(1)
        elif args.algo_palette:
            selected_palette = get_algorithmic_palette(image, args.algo_palette, args.colors)
        else:
            selected_palette = generate_median_cut_palette(image, args.colors)

        try:
            dither_mode = DitherMode(args.mode)
        except ValueError:
            print(f"Error: Unsupported Dither Mode '{args.mode}'.")
            sys.exit(1)

        ditherer = ImageDitherer(
            num_colors=args.colors,
            dither_mode=dither_mode,
            palette=selected_palette
        )

        try:
            dithered_image = ditherer.apply_dithering(image)
        except Exception as e:
            print(f"Error: Failed to apply dithering: {e}")
            sys.exit(1)

        try:
            dithered_image.save(args.output_image)
            print(f"Success: Dithered image saved to '{args.output_image}'.")
        except Exception as e:
            print(f"Error: Failed to save image '{args.output_image}': {e}")
            sys.exit(1)

    elif args.command == 'dither-pixelize':
        try:
            image = Image.open(args.input_image)
        except Exception as e:
            print(f"Error: Failed to open image '{args.input_image}': {e}")
            sys.exit(1)

        try:
            pixelized_image = pixelize_image(image, args.max_size)
        except Exception as e:
            print(f"Error: Failed to pixelize image: {e}")
            sys.exit(1)

        if args.algo_palette and args.palette:
            print("Error: You cannot specify both --algo-palette and --palette. Choose one.")
            sys.exit(1)

        if args.palette:
            selected_palette = None
            found_custom = False
            for name, pal in custom_palettes:
                if name.lower() == args.palette.lower():
                    selected_palette = pal
                    found_custom = True
                    break
            if not found_custom:
                print(f"Error: Custom palette '{args.palette}' not found in palette.json.")
                sys.exit(1)
        elif args.algo_palette:
            selected_palette = get_algorithmic_palette(pixelized_image, args.algo_palette, args.colors)
        else:
            selected_palette = generate_median_cut_palette(pixelized_image, args.colors)

        try:
            dither_mode = DitherMode(args.mode)
        except ValueError:
            print(f"Error: Unsupported Dither Mode '{args.mode}'.")
            sys.exit(1)

        ditherer = ImageDitherer(
            num_colors=args.colors,
            dither_mode=dither_mode,
            palette=selected_palette
        )

        try:
            dithered_image = ditherer.apply_dithering(pixelized_image)
        except Exception as e:
            print(f"Error: Failed to apply dithering: {e}")
            sys.exit(1)

        try:
            dithered_image.save(args.output_image)
            print(f"Success: Pixelized and dithered image saved to '{args.output_image}'.")
        except Exception as e:
            print(f"Error: Failed to save image '{args.output_image}': {e}")
            sys.exit(1)

    else:
        print("Error: Unknown command.")
        parser.print_help()
        sys.exit(1)

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex code to RGB tuple"""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# -------------------- Main Function --------------------

def main():
    import sys
    if len(sys.argv) > 1:
        run_cli()
    else:
        app = App()
        app.mainloop()

if __name__ == "__main__":
    main()
