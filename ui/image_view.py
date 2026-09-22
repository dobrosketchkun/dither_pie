"""Image view components."""

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

__all__ = ['ZoomableImage', 'ImageComparisonView']


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
        self._auto_fit_on_resize = True

        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<MouseWheel>", self.zoom)
        # For Linux
        self.bind("<Button-4>", self.zoom)
        self.bind("<Button-5>", self.zoom)
        self.bind("<Configure>", self.on_resize)

    def set_image(self, image: Image.Image, update: bool = True):
        """
        Set the image to display.
        
        Args:
            image: PIL Image to display
            update: If True, immediately update the view. Set to False if you'll call fit_to_window() right after.
        """
        self.original_image = image
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._auto_fit_on_resize = True
        if update:
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
        self._auto_fit_on_resize = True
        self.update_view()

    # Transparency checkerboard, in screen pixels so squares stay a constant
    # size however far the user zooms in.
    _CHECKER_SQUARE = 8
    _CHECKER_LIGHT = (99, 99, 99)
    _CHECKER_DARK = (72, 72, 72)

    @classmethod
    def _make_checkerboard(cls, w: int, h: int) -> Image.Image:
        """Build a w x h checkerboard by tiling a 2x2-square patch."""
        s = cls._CHECKER_SQUARE
        tile = np.empty((2 * s, 2 * s, 3), dtype=np.uint8)
        tile[:s, :s] = cls._CHECKER_LIGHT
        tile[s:, s:] = cls._CHECKER_LIGHT
        tile[:s, s:] = cls._CHECKER_DARK
        tile[s:, :s] = cls._CHECKER_DARK
        reps = ((h + 2 * s - 1) // (2 * s), (w + 2 * s - 1) // (2 * s), 1)
        return Image.fromarray(np.tile(tile, reps)[:h, :w], 'RGB')

    @classmethod
    def _composite_for_display(cls, image: Image.Image) -> Image.Image:
        """
        Flatten a transparent image onto a checkerboard for display.

        Without this, transparent regions take the canvas background and read
        as black content rather than as empty space.
        """
        if image.mode not in ('RGBA', 'LA', 'PA', 'P'):
            return image
        rgba = image.convert('RGBA')
        if rgba.getchannel('A').getextrema()[0] == 255:
            return image  # nothing actually transparent
        backdrop = cls._make_checkerboard(rgba.width, rgba.height).convert('RGBA')
        return Image.alpha_composite(backdrop, rgba).convert('RGB')

    def update_view(self):
        if not self.original_image:
            return
        nw = int(self.original_image.width * self.zoom_factor)
        nh = int(self.original_image.height * self.zoom_factor)
        if nw <= 0 or nh <= 0:
            return

        resized = self.original_image.resize((nw, nh), Image.Resampling.NEAREST)
        resized = self._composite_for_display(resized)
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
        self._auto_fit_on_resize = False

    def pan(self, event):
        self.offset_x = event.x - self.pan_start_x
        self.offset_y = event.y - self.pan_start_y
        self._auto_fit_on_resize = False
        self.update_view()

    def zoom(self, event):
        if not self.original_image:
            return
        fine = False
        if hasattr(event, "state"):
            fine = bool(event.state & 0x0001)  # Shift key
        step = 0.98 if fine else 0.9
        grow = 1.02 if fine else 1.1
        # Zoom out on negative delta or Button-5
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_factor *= step
        else:
            self.zoom_factor *= grow
        self.zoom_factor = max(0.01, min(30.0, self.zoom_factor))
        self._auto_fit_on_resize = False
        self.update_view()

    def on_resize(self, event):
        if self._auto_fit_on_resize:
            self.fit_to_window()
        else:
            self.update_view()


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
