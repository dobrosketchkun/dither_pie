"""
Reusable GUI components for the dithering application.
Separated from main app for better maintainability.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from typing import List, Tuple, Optional, Callable


class ZoomableImage(tk.Canvas):
    """
    A custom widget that supports zooming & panning for a displayed PIL image.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.original_image = None
        self.displayed_image = None
        self.image_id = None
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0

        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<MouseWheel>", self.zoom)
        # For Linux
        self.bind("<Button-4>", self.zoom)
        self.bind("<Button-5>", self.zoom)
        self.bind("<Configure>", self.on_resize)

    def set_image(self, image: Image.Image):
        self.original_image = image
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_view()

    def fit_to_window(self):
        if not self.original_image:
            return
        self.update_idletasks()
        cw = self.winfo_width()
        ch = self.winfo_height()
        iw, ih = self.original_image.size
        if iw == 0 or ih == 0:
            return
        wr = cw / iw
        hr = ch / ih
        self.zoom_factor = min(wr, hr)
        self.offset_x = 0
        self.offset_y = 0
        self.update_view()

    def update_view(self):
        if not self.original_image:
            return
        nw = int(self.original_image.width * self.zoom_factor)
        nh = int(self.original_image.height * self.zoom_factor)
        if nw <= 0 or nh <= 0:
            return

        resized = self.original_image.resize((nw, nh), Image.Resampling.NEAREST)
        self.displayed_image = ImageTk.PhotoImage(resized)

        cw = self.winfo_width()
        ch = self.winfo_height()
        x = (cw - nw)//2 + self.offset_x
        y = (ch - nh)//2 + self.offset_y

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
        # Zoom out on negative delta or Button-5
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_factor *= 0.9
        else:
            self.zoom_factor *= 1.1
        self.zoom_factor = max(0.01, min(30.0, self.zoom_factor))
        self.update_view()

    def on_resize(self, event):
        self.fit_to_window()


class PalettePreview(ctk.CTkFrame):
    """
    Shows a small horizontal bar for each color in the palette.
    """
    def __init__(self, master, palette, width=200, height=30, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.palette = palette
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.after(100, self.draw_palette)
        self.bind("<Configure>", lambda ev: self.after(100, self.draw_palette))

    def draw_palette(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        n = len(self.palette)
        if n == 0:
            return
        seg_w = w / n
        for i, color in enumerate(self.palette):
            x1 = i*seg_w
            x2 = (i+1)*seg_w
            hx = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            self.canvas.create_rectangle(x1, 0, x2, h, fill=hx, outline='')


class ProgressDialog(ctk.CTkToplevel):
    """
    Modal dialog showing progress bar and status message.
    """
    def __init__(self, parent, title="Processing"):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x150")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (500 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (150 // 2)
        self.geometry(f"+{x}+{y}")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=("Arial", 14)
        )
        self.status_label.pack(pady=(20, 10))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Percentage label
        self.percent_label = ctk.CTkLabel(
            self,
            text="0%",
            font=("Arial", 12)
        )
        self.percent_label.pack(pady=5)
        
        # Cancel button
        self.cancelled = False
        self.cancel_button = ctk.CTkButton(
            self,
            text="Cancel",
            command=self.cancel,
            width=100
        )
        self.cancel_button.pack(pady=10)
        
        self.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def update_progress(self, fraction: float, message: str = ""):
        """Update progress bar and status message."""
        self.progress_bar.set(fraction)
        self.percent_label.configure(text=f"{int(fraction * 100)}%")
        if message:
            self.status_label.configure(text=message)
        self.update()
    
    def cancel(self):
        """Mark as cancelled and close."""
        self.cancelled = True
        self.destroy()
    
    def is_cancelled(self) -> bool:
        """Check if user cancelled the operation."""
        return self.cancelled


class ColorPickerGrid(ctk.CTkFrame):
    """
    Grid of color swatches that can be clicked to edit.
    """
    def __init__(self, master, colors: List[Tuple[int, int, int]], 
                 on_color_change: Callable[[int, Tuple[int, int, int]], None],
                 **kwargs):
        super().__init__(master, **kwargs)
        self.colors = colors
        self.on_color_change = on_color_change
        self.color_buttons = []
        
        self.build_grid()
    
    def build_grid(self):
        """Build the grid of color swatches."""
        cols = min(8, len(self.colors))
        rows = (len(self.colors) + cols - 1) // cols
        
        for i, color in enumerate(self.colors):
            row = i // cols
            col = i % cols
            
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            btn = tk.Button(
                self,
                bg=hex_color,
                width=4,
                height=2,
                relief="raised",
                bd=2,
                command=lambda idx=i, c=color: self.edit_color(idx, c)
            )
            btn.grid(row=row, column=col, padx=2, pady=2)
            self.color_buttons.append(btn)
    
    def edit_color(self, index: int, current_color: Tuple[int, int, int]):
        """Open color picker to edit a color."""
        from tkinter import colorchooser
        
        hex_color = f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}'
        result = colorchooser.askcolor(
            color=hex_color,
            title=f"Edit Color {index + 1}",
            parent=self
        )
        
        if result and result[0]:
            # result[0] is RGB tuple as floats
            new_color = tuple(int(c) for c in result[0])
            self.colors[index] = new_color
            
            # Update button color
            hex_new = f'#{new_color[0]:02x}{new_color[1]:02x}{new_color[2]:02x}'
            self.color_buttons[index].configure(bg=hex_new)
            
            # Notify callback
            self.on_color_change(index, new_color)
    
    def get_colors(self) -> List[Tuple[int, int, int]]:
        """Get current color list."""
        return self.colors


class StatusBar(ctk.CTkFrame):
    """
    Status bar to show information at the bottom of the window.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, height=30, **kwargs)
        
        self.label = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w"
        )
        self.label.pack(side="left", padx=10, fill="x", expand=True)
    
    def set_status(self, message: str):
        """Update status message."""
        self.label.configure(text=message)
        self.update()


class ImageComparisonView(ctk.CTkFrame):
    """
    Side-by-side comparison of two images with synchronized zooming.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        # Labels
        self.left_label = ctk.CTkLabel(self, text="Original")
        self.left_label.grid(row=0, column=0, pady=5)
        
        self.right_label = ctk.CTkLabel(self, text="Processed")
        self.right_label.grid(row=0, column=1, pady=5)
        
        # Image viewers
        self.left_viewer = ZoomableImage(self, bg="gray20", highlightthickness=0)
        self.left_viewer.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        self.right_viewer = ZoomableImage(self, bg="gray20", highlightthickness=0)
        self.right_viewer.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
    
    def set_images(self, left_image: Image.Image, right_image: Image.Image):
        """Set the images to display."""
        self.left_viewer.set_image(left_image)
        self.right_viewer.set_image(right_image)

