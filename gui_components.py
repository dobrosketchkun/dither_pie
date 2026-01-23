"""
Reusable GUI components for the dithering application.
Separated from main app for better maintainability.
"""

import os
import json
import colorsys
import ctypes
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from typing import List, Tuple, Optional, Callable

__all__ = [
    'ZoomableImage',
    'PixelizationEditorCanvas',
    'PalettePreview',
    'ProgressDialog',
    'ColorPickerGrid',
    'StatusBar',
    'ImageComparisonView',
    'HSVColorPickerDialog',
    'CustomPaletteCreator',
    'PaletteImagePreviewDialog',
    'DitherSettingsDialog',
    'PixelizationEditorDialog',
]


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


class PixelizationEditorCanvas(ZoomableImage):
    """
    Canvas with grid overlay and pixel-editing tools.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.source_image = None
        self.grid_w = 1
        self.grid_h = 1
        self.show_grid = True
        self.mode = "preview"  # "preview" or "edit"
        self.tool = "brush"  # "brush", "magic", "picker"
        self.tool_size = 1
        self.magic_threshold = 5
        self.draw_color = (0, 0, 0)
        self.highlight_cell = None
        self.pixel_colors = []
        self.history = []
        self.redo = []
        self._drawing_active = False
        self._drawing_occurred = False
        self._last_draw_cell = None
        self.on_color_pick = None
        self.preview_grid_scale = 1.0
        self.alt_zoom_active = False
        self.preview_grid_offset_x = 0.0
        self.preview_grid_offset_y = 0.0
        self._last_pan_event = None

        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        self.bind("<ButtonPress-3>", self._on_right_down)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_right_up)
        self.bind("<Motion>", self._on_motion)

    def set_mode(self, mode: str):
        self.mode = mode
        self.highlight_cell = None
        self.update_view()

    def set_source_image(self, image: Image.Image):
        self.source_image = image

    def set_grid(self, grid_w: int, grid_h: int):
        self.grid_w = max(1, int(grid_w))
        self.grid_h = max(1, int(grid_h))
        self.preview_grid_scale = 1.0
        self.preview_grid_offset_x = 0.0
        self.preview_grid_offset_y = 0.0
        if self.mode == "edit":
            self._ensure_pixel_data()
        self.update_view()

    def set_tool(self, tool: str):
        self.tool = tool

    def set_tool_size(self, size: int):
        self.tool_size = max(1, int(size))
        self.update_view()

    def set_draw_color(self, color: Tuple[int, int, int]):
        self.draw_color = color

    def set_magic_threshold(self, value: int):
        self.magic_threshold = max(0, int(value))

    def set_show_grid(self, show: bool):
        self.show_grid = bool(show)
        self.update_view()

    def set_pixel_data(self, pixel_colors: List[List[Optional[Tuple[int, int, int]]]]):
        self.pixel_colors = pixel_colors
        self._reset_history()
        self._update_image_from_pixels(preserve_view=False)

    def get_pixel_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        return self.pixel_colors

    def undo(self):
        if len(self.history) <= 1:
            return
        state = self.history.pop()
        self.redo.append(state)
        self.pixel_colors = self._deep_copy_pixels(self.history[-1])
        self._update_image_from_pixels(preserve_view=True)

    def redo_action(self):
        if not self.redo:
            return
        state = self.redo.pop()
        self.history.append(self._deep_copy_pixels(state))
        self.pixel_colors = self._deep_copy_pixels(state)
        self._update_image_from_pixels(preserve_view=True)

    def update_view(self):
        super().update_view()
        self._draw_overlays()

    def _get_image_draw_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.original_image:
            return None
        nw = int(self.original_image.width * self.zoom_factor)
        nh = int(self.original_image.height * self.zoom_factor)
        if nw <= 0 or nh <= 0:
            return None
        cw = self.winfo_width()
        ch = self.winfo_height()
        x = (cw - nw) // 2 + self.offset_x
        y = (ch - nh) // 2 + self.offset_y
        return x, y, nw, nh

    def _get_image_transform(self) -> Optional[Tuple[int, int, float]]:
        rect = self._get_image_draw_rect()
        if not rect:
            return None
        x, y, _, _ = rect
        return x, y, self.zoom_factor

    def _get_preview_grid_rect(self) -> Optional[Tuple[float, float, float, float, float]]:
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            return None
        base_cell = min(cw / self.grid_w, ch / self.grid_h)
        cell = base_cell * self.preview_grid_scale
        if cell <= 0:
            return None
        gw = self.grid_w * cell
        gh = self.grid_h * cell
        x0 = (cw - gw) / 2 + self.preview_grid_offset_x
        y0 = (ch - gh) / 2 + self.preview_grid_offset_y
        return x0, y0, gw, gh, cell

    def fit_image_to_grid(self):
        if not self.original_image:
            return
        self.preview_grid_scale = 1.0
        self.preview_grid_offset_x = 0.0
        self.preview_grid_offset_y = 0.0
        self._auto_fit_on_resize = True
        grid_rect = self._get_preview_grid_rect()
        if not grid_rect:
            return
        x0, y0, gw, gh, _ = grid_rect
        iw, ih = self.original_image.size
        if iw == 0 or ih == 0:
            return
        self.zoom_factor = min(gw / iw, gh / ih)
        nw = iw * self.zoom_factor
        nh = ih * self.zoom_factor
        cw = self.winfo_width()
        ch = self.winfo_height()
        target_x = x0 + (gw - nw) / 2
        target_y = y0 + (gh - nh) / 2
        self.offset_x = target_x - (cw - nw) / 2
        self.offset_y = target_y - (ch - nh) / 2
        self.update_view()

    @staticmethod
    def _shift_down(event) -> bool:
        if hasattr(event, "state"):
            return bool(event.state & 0x0001)
        return False

    def start_pan(self, event):
        self._last_pan_event = (event.x, event.y)
        super().start_pan(event)

    def pan(self, event):
        if self.mode == "preview" and self._shift_down(event):
            if self._last_pan_event:
                dx = event.x - self._last_pan_event[0]
                dy = event.y - self._last_pan_event[1]
                self.preview_grid_offset_x += dx
                self.preview_grid_offset_y += dy
        self._last_pan_event = (event.x, event.y)
        super().pan(event)

    def on_resize(self, event):
        if self._auto_fit_on_resize:
            if self.mode == "preview":
                self.fit_image_to_grid()
            else:
                self.fit_to_window()
        else:
            self.update_view()

    def zoom(self, event):
        if not self.original_image:
            return
        fine = False
        if hasattr(event, "state"):
            fine = bool(event.state & 0x0001)  # Shift key
        step = 0.98 if fine else 0.9
        grow = 1.02 if fine else 1.1
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = step
        else:
            factor = grow
        self.zoom_factor *= factor
        self.zoom_factor = max(0.01, min(30.0, self.zoom_factor))
        if self.alt_zoom_active and self.mode == "preview":
            self.preview_grid_scale *= factor
            self.preview_grid_scale = max(0.1, min(10.0, self.preview_grid_scale))
        self._auto_fit_on_resize = False
        self.update_view()

    def _draw_overlays(self):
        self.delete("grid")
        self.delete("highlight")
        if self.grid_w <= 0 or self.grid_h <= 0:
            return
        if self.mode == "preview":
            self._draw_preview_grid()
        else:
            self._draw_edit_grid()
            self._draw_highlight()

    def _draw_preview_grid(self):
        if not self.show_grid:
            return
        grid_rect = self._get_preview_grid_rect()
        if not grid_rect:
            return
        x0, y0, nw, nh, cell_w = grid_rect
        cell_h = cell_w
        if min(cell_w, cell_h) < 3:
            return
        for i in range(1, self.grid_w):
            x = x0 + i * cell_w
            self.create_line(x, y0, x, y0 + nh, fill="#ffffff", width=1, tags="grid", stipple="gray50")
        for j in range(1, self.grid_h):
            y = y0 + j * cell_h
            self.create_line(x0, y, x0 + nw, y, fill="#ffffff", width=1, tags="grid", stipple="gray50")

    def _draw_edit_grid(self):
        if not self.show_grid:
            return
        rect = self._get_image_draw_rect()
        if not rect:
            return
        x0, y0, _, _ = rect
        if self.zoom_factor < 3:
            return
        for i in range(1, self.grid_w):
            x = x0 + i * self.zoom_factor
            self.create_line(x, y0, x, y0 + self.grid_h * self.zoom_factor,
                             fill="#ffffff", width=1, tags="grid", stipple="gray50")
        for j in range(1, self.grid_h):
            y = y0 + j * self.zoom_factor
            self.create_line(x0, y, x0 + self.grid_w * self.zoom_factor, y,
                             fill="#ffffff", width=1, tags="grid", stipple="gray50")

    def _draw_highlight(self):
        if self.highlight_cell is None:
            return
        rect = self._get_image_draw_rect()
        if not rect:
            return
        if self.tool == "picker":
            return
        x0, y0, _, _ = rect
        i, j = self.highlight_cell
        size = self.tool_size
        x = x0 + i * self.zoom_factor
        y = y0 + j * self.zoom_factor
        w = size * self.zoom_factor
        h = size * self.zoom_factor
        self.create_rectangle(x, y, x + w, y + h, outline="#ff3333",
                              width=2, tags="highlight")

    def _ensure_pixel_data(self):
        if self.pixel_colors:
            return
        self.pixel_colors = [
            [None for _ in range(self.grid_w)]
            for _ in range(self.grid_h)
        ]
        self._reset_history()

    def _reset_history(self):
        if not self.pixel_colors:
            return
        self.history = [self._deep_copy_pixels(self.pixel_colors)]
        self.redo = []

    def _push_history(self):
        self.history.append(self._deep_copy_pixels(self.pixel_colors))
        self.redo = []

    @staticmethod
    def _deep_copy_pixels(pixels):
        return [row[:] for row in pixels]

    def _update_image_from_pixels(self, preserve_view: bool):
        if not self.pixel_colors:
            return
        img = Image.new("RGBA", (self.grid_w, self.grid_h), (0, 0, 0, 0))
        px = img.load()
        for j in range(self.grid_h):
            for i in range(self.grid_w):
                color = self.pixel_colors[j][i]
                if color is None:
                    continue
                px[i, j] = (*color, 255)
        if preserve_view:
            self._set_image_preserve_view(img)
        else:
            self.set_image(img, update=True)

    def _set_image_preserve_view(self, image: Image.Image):
        zoom = self.zoom_factor
        ox = self.offset_x
        oy = self.offset_y
        self.set_image(image, update=False)
        self.zoom_factor = zoom
        self.offset_x = ox
        self.offset_y = oy
        self.update_view()

    def _canvas_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        rect = self._get_image_draw_rect()
        if not rect:
            return None
        x0, y0, _, _ = rect
        img_x = (x - x0) / self.zoom_factor
        img_y = (y - y0) / self.zoom_factor
        i = int(img_x)
        j = int(img_y)
        if i < 0 or j < 0 or i >= self.grid_w or j >= self.grid_h:
            return None
        return i, j

    def _on_left_down(self, event):
        if self.mode == "preview":
            if self.tool == "picker":
                self._pick_color_at_canvas(event.x, event.y)
                self.update_view()
                return
            self.start_pan(event)
            return
        self._ensure_pixel_data()
        cell = self._canvas_to_cell(event.x, event.y)
        if not cell:
            return
        if self.tool == "picker":
            self._pick_color_at_canvas(event.x, event.y)
            self.update_view()
            return
        if self.tool == "magic":
            self._apply_magic_wand(cell)
            self._push_history()
            self.update_view()
        else:
            self._drawing_active = True
            self._last_draw_cell = cell
            self._apply_brush(cell)
            self._drawing_occurred = True
            self.update_view()

    def _on_left_drag(self, event):
        if self.mode == "preview":
            self.pan(event)
            return
        if not self._drawing_active:
            return
        cell = self._canvas_to_cell(event.x, event.y)
        if cell:
            if self._last_draw_cell is None:
                self._last_draw_cell = cell
            self._apply_brush_line(self._last_draw_cell, cell)
            self._last_draw_cell = cell
            self._drawing_occurred = True
            self.update_view()

    def _on_left_up(self, _event):
        if self.mode == "preview":
            self._last_pan_event = None
            return
        if self._drawing_active:
            self._drawing_active = False
            self._last_draw_cell = None
            if self._drawing_occurred:
                self._push_history()
                self._drawing_occurred = False

    def _on_right_down(self, event):
        self.start_pan(event)

    def _on_right_drag(self, event):
        self.pan(event)

    def _on_right_up(self, _event):
        self._last_pan_event = None
        return

    def _on_motion(self, event):
        if self.mode != "edit":
            return
        cell = self._canvas_to_cell(event.x, event.y)
        if cell != self.highlight_cell:
            self.highlight_cell = cell
            self.update_view()

    def _apply_brush(self, cell: Tuple[int, int]):
        i0, j0 = cell
        if self.tool == "brush":
            color = self.draw_color
        else:
            return
        size = self.tool_size
        for dj in range(size):
            for di in range(size):
                i = min(self.grid_w - 1, i0 + di)
                j = min(self.grid_h - 1, j0 + dj)
                self.pixel_colors[j][i] = color
        self._update_image_from_pixels(preserve_view=True)

    def _apply_brush_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self._apply_brush((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _apply_magic_wand(self, cell: Tuple[int, int]):
        i0, j0 = cell
        size = max(1, int(self.tool_size))
        start_i = min(self.grid_w - 1, i0)
        start_j = min(self.grid_h - 1, j0)
        end_i = min(self.grid_w - 1, i0 + size - 1)
        end_j = min(self.grid_h - 1, j0 + size - 1)
        targets = []
        for j in range(start_j, end_j + 1):
            for i in range(start_i, end_i + 1):
                color = self.pixel_colors[j][i]
                if color is None:
                    continue
                if color not in targets:
                    targets.append(color)
        for target in targets:
            start_cell = self._find_cell_with_color(start_i, start_j, end_i, end_j, target)
            if start_cell:
                self._flood_fill_from(start_cell, target)
        self._update_image_from_pixels(preserve_view=True)

    def _find_cell_with_color(self, start_i, start_j, end_i, end_j, target):
        for j in range(start_j, end_j + 1):
            for i in range(start_i, end_i + 1):
                color = self.pixel_colors[j][i]
                if color is not None and color == target:
                    return (i, j)
        return None

    def _flood_fill_from(self, cell: Tuple[int, int], target: Tuple[int, int, int]):
        i, j = cell
        threshold = self.magic_threshold
        stack = [(i, j)]
        visited = set()
        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            visited.add((ci, cj))
            if ci < 0 or cj < 0 or ci >= self.grid_w or cj >= self.grid_h:
                continue
            color = self.pixel_colors[cj][ci]
            if color is None:
                continue
            if self._color_distance(color, target) > threshold:
                continue
            self.pixel_colors[cj][ci] = None
            stack.extend([
                (ci - 1, cj), (ci + 1, cj),
                (ci, cj - 1), (ci, cj + 1)
            ])

    def _pick_color_at_canvas(self, x: float, y: float):
        color = None
        if self.mode == "edit" and self.pixel_colors:
            cell = self._canvas_to_cell(x, y)
            if cell:
                i, j = cell
                color = self.pixel_colors[j][i]
        if color is None and self.source_image is not None:
            transform = self._get_image_transform()
            if transform:
                origin_x, origin_y, scale = transform
                ox = int((x - origin_x) / scale)
                oy = int((y - origin_y) / scale)
                ox = max(0, min(self.source_image.width - 1, ox))
                oy = max(0, min(self.source_image.height - 1, oy))
                color = self.source_image.getpixel((ox, oy))[:3]
        if color is None:
            return
        self.draw_color = color
        if self.on_color_pick:
            self.on_color_pick(color)

    @staticmethod
    def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        dr = c1[0] - c2[0]
        dg = c1[1] - c2[1]
        db = c1[2] - c2[2]
        return (dr * dr + dg * dg + db * db) ** 0.5


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
    Supports animated spinners for processing states.
    """
    def __init__(self, master, spinner_name: str = "dots", **kwargs):
        super().__init__(master, height=30, **kwargs)
        
        self.label = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w"
        )
        self.label.pack(side="left", padx=10, fill="x", expand=True)
        
        # Spinner state
        self.spinner_active = False
        self.spinner_frames = []
        self.spinner_interval = 80
        self.spinner_index = 0
        self.spinner_message = ""
        self.spinner_after_id = None
        
        # Load spinner configuration
        self._load_spinner(spinner_name)
    
    def _load_spinner(self, spinner_name: str):
        """Load spinner configuration from spinners.json."""
        try:
            import json
            from pathlib import Path
            
            # Look for spinners.json in the same directory as this file
            spinner_file = Path(__file__).parent / "spinners.json"
            if spinner_file.exists():
                with open(spinner_file, 'r', encoding='utf-8') as f:
                    spinners = json.load(f)
                    if spinner_name in spinners:
                        spinner_config = spinners[spinner_name]
                        self.spinner_frames = spinner_config.get('frames', ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
                        self.spinner_interval = spinner_config.get('interval', 80)
                    else:
                        # Default fallback
                        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                        self.spinner_interval = 80
            else:
                # Default fallback if file not found
                self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                self.spinner_interval = 80
        except Exception as e:
            print(f"Error loading spinner: {e}")
            # Simple fallback
            self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            self.spinner_interval = 80
    
    def set_status(self, message: str, spinning: bool = False):
        """
        Update status message.
        
        Args:
            message: The status message to display
            spinning: If True, starts animated spinner. If False, stops any active spinner.
        """
        if spinning:
            self.start_spinner(message)
        else:
            self.stop_spinner()
            self.label.configure(text=message)
            self.update()
    
    def start_spinner(self, message: str):
        """Start the animated spinner with the given message."""
        self.spinner_message = message
        self.spinner_active = True
        self.spinner_index = 0
        self._animate_spinner()
    
    def stop_spinner(self):
        """Stop the animated spinner."""
        self.spinner_active = False
        if self.spinner_after_id is not None:
            self.after_cancel(self.spinner_after_id)
            self.spinner_after_id = None
    
    def _animate_spinner(self):
        """Internal method to animate the spinner."""
        if not self.spinner_active:
            return
        
        frame = self.spinner_frames[self.spinner_index]
        self.label.configure(text=f"{frame} {self.spinner_message}")
        self.update()
        
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        self.spinner_after_id = self.after(self.spinner_interval, self._animate_spinner)


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


# -------------------- HSV Color Picker Dialog --------------------

PLANE_SIZE = 256

class HSVColorPickerDialog(ctk.CTkToplevel):
    """
    A simple HSV color picker used for building custom palettes.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("HSV Color Picker")
        self.geometry("640x400")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.hue = 0.0
        self.sat = 0.0
        self.val = 1.0
        self.selected_color = None

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Hue gradient
        self.hue_gradient_image = self.create_hue_gradient(width=360, height=20)
        self.hue_gradient_photo = ImageTk.PhotoImage(self.hue_gradient_image)
        self.hue_gradient_label = tk.Label(
            main_frame, image=self.hue_gradient_photo, bd=1, relief='ridge'
        )
        self.hue_gradient_label.grid(row=0, column=0, columnspan=2,
                                     pady=(0,5), sticky='w')

        self.hue_slider = ctk.CTkSlider(
            main_frame, from_=0, to=360, command=self.on_hue_changed, width=360
        )
        self.hue_slider.set(0)
        self.hue_slider.grid(row=1, column=0, columnspan=2, padx=0, pady=(0,10), sticky='w')

        # Color plane (S-V plane)
        self.plane_canvas = tk.Canvas(
            main_frame, width=PLANE_SIZE, height=PLANE_SIZE,
            bd=2, relief='sunken', cursor='cross'
        )
        self.plane_canvas.grid(row=2, column=0, padx=(0,10), pady=5)
        self.plane_canvas.bind("<Button-1>", self.on_plane_click)
        self.plane_canvas.bind("<B1-Motion>", self.on_plane_click)
        self.plane_image = None
        self.plane_photo = None
        self.circle_id = None

        # Right side frame for numeric entry & preview
        self.create_color_representations(main_frame)

        ok_button = ctk.CTkButton(self, text="OK", command=self.on_ok)
        ok_button.pack(side='bottom', pady=(0,10))

        self.update_color_plane()
        self.update_preview()

    def create_hue_gradient(self, width=360, height=20):
        img = Image.new("RGB", (width, height), "black")
        for x in range(width):
            hue_norm = x / float(width)
            h = hue_norm * 360
            r, g, b = colorsys.hsv_to_rgb(h/360.0, 1.0, 1.0)
            for y in range(height):
                img.putpixel((x,y), (int(r*255), int(g*255), int(b*255)))
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

        self.sat = x / float(PLANE_SIZE-1)
        self.val = 1.0 - (y / float(PLANE_SIZE-1))
        self.update_preview()
        self.update_circle()
        self.update_color_reps()

    def update_color_plane(self):
        img = Image.new("RGB", (PLANE_SIZE, PLANE_SIZE), "black")
        hue_norm = self.hue / 360.0
        for j in range(PLANE_SIZE):
            v = 1.0 - j / float(PLANE_SIZE-1)
            for i in range(PLANE_SIZE):
                s = i / float(PLANE_SIZE-1)
                r, g, b = colorsys.hsv_to_rgb(hue_norm, s, v)
                img.putpixel((i,j), (int(r*255), int(g*255), int(b*255)))
        self.plane_image = img
        self.plane_photo = ImageTk.PhotoImage(img)
        self.plane_canvas.create_image(0, 0, anchor='nw', image=self.plane_photo)
        self.update_circle()

    def update_preview(self):
        r,g,b = self.get_rgb()
        hx = f"#{r:02x}{g:02x}{b:02x}"
        if hasattr(self, 'preview_box'):
            self.preview_box.configure(fg_color=hx)

    def update_circle(self):
        if self.plane_photo is None:
            return
        if self.circle_id is not None:
            self.plane_canvas.delete(self.circle_id)
            self.circle_id = None

        x = self.sat*(PLANE_SIZE-1)
        y = (1.0-self.val)*(PLANE_SIZE-1)
        rad = 5
        x0 = x - rad
        y0 = y - rad
        x1 = x + rad
        y1 = y + rad
        try:
            bgc = self.plane_image.getpixel((int(x), int(y)))
            lum = 0.2126*bgc[0] + 0.7152*bgc[1] + 0.0722*bgc[2]
            if lum > 128:
                oc = "#000000"
            else:
                oc = "#FFFFFF"
        except:
            oc = "#FFFFFF"

        self.circle_id = self.plane_canvas.create_oval(x0, y0, x1, y1, outline=oc, width=2)

    def create_color_representations(self, parent):
        rf = ctk.CTkFrame(parent)
        rf.grid(row=2, column=1, padx=10, pady=5, sticky='n')

        rgb_lab = ctk.CTkLabel(rf, text="RGB:")
        rgb_lab.grid(row=0, column=0, padx=5, pady=(0,5), sticky='w')

        self.r_var = tk.StringVar(value="255")
        self.g_var = tk.StringVar(value="255")
        self.b_var = tk.StringVar(value="255")

        self.r_entry = ctk.CTkEntry(rf, textvariable=self.r_var, width=60)
        self.g_entry = ctk.CTkEntry(rf, textvariable=self.g_var, width=60)
        self.b_entry = ctk.CTkEntry(rf, textvariable=self.b_var, width=60)

        self.r_entry.bind("<Return>", self.on_rgb_enter)
        self.g_entry.bind("<Return>", self.on_rgb_enter)
        self.b_entry.bind("<Return>", self.on_rgb_enter)

        self.r_entry.grid(row=0, column=1, padx=5, pady=(0,5))
        self.g_entry.grid(row=0, column=2, padx=5, pady=(0,5))
        self.b_entry.grid(row=0, column=3, padx=5, pady=(0,5))

        hex_lab = ctk.CTkLabel(rf, text="HEX:")
        hex_lab.grid(row=1, column=0, padx=5, pady=(10,5), sticky='w')

        self.hex_var = tk.StringVar(value="#FFFFFF")
        self.hex_entry = ctk.CTkEntry(rf, textvariable=self.hex_var, width=180)
        self.hex_entry.bind("<Return>", self.on_hex_enter)
        self.hex_entry.grid(row=1, column=1, columnspan=3, padx=(5,0), pady=(10,5), sticky='w')

        prev_lab = ctk.CTkLabel(rf, text="Selected Color:")
        prev_lab.grid(row=2, column=0, padx=5, pady=(10,5), sticky='w')

        self.preview_box = ctk.CTkLabel(rf, text="", width=80, height=40, fg_color="#ffffff", corner_radius=6)
        self.preview_box.grid(row=2, column=1, padx=5, pady=(10,5), sticky='w')

    def get_rgb(self):
        r,g,b = colorsys.hsv_to_rgb(self.hue/360.0, self.sat, self.val)
        return int(r*255), int(g*255), int(b*255)

    def update_color_reps(self):
        r,g,b = self.get_rgb()
        self.r_var.set(str(r))
        self.g_var.set(str(g))
        self.b_var.set(str(b))
        self.hex_var.set(f"#{r:02x}{g:02x}{b:02x}")

    def on_rgb_enter(self, event):
        try:
            r = int(self.r_var.get())
            g = int(self.g_var.get())
            b = int(self.b_var.get())
            if r<0 or g<0 or b<0 or r>255 or g>255 or b>255:
                raise ValueError("RGB must be [0..255]")
            h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            self.hue = h*360
            self.sat = s
            self.val = v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def on_hex_enter(self, event):
        try:
            hx = self.hex_var.get().strip()
            if not hx.startswith('#') or len(hx) != 7:
                raise ValueError("HEX code must be #RRGGBB.")
            r = int(hx[1:3], 16)
            g = int(hx[3:5], 16)
            b = int(hx[5:7], 16)
            h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            self.hue = h*360
            self.sat = s
            self.val = v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def on_ok(self):
        self.selected_color = self.get_rgb()
        self.destroy()


# -------------------- Custom Palette Creator --------------------

class CustomPaletteCreator(ctk.CTkToplevel):
    """
    A small editor that lets you create a custom palette by picking multiple HSV colors.
    """
    def __init__(self, parent, palette_manager, refresh_callback):
        super().__init__(parent)
        self.title("Create Custom Palette")
        self.geometry("500x400")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.palette_manager = palette_manager
        self.refresh_callback = refresh_callback
        self.colors = []

        self.palette_frame = ctk.CTkFrame(self)
        self.palette_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.update_palette_display()

        self.save_button = ctk.CTkButton(self, text="Save Palette", command=self.save_palette)
        self.save_button.pack(pady=10)

    def update_palette_display(self):
        for w in self.palette_frame.winfo_children():
            w.destroy()

        square_size = 40
        pad = 5
        for idx, col in enumerate(self.colors):
            hx = f'#{col[0]:02x}{col[1]:02x}{col[2]:02x}'
            btn = tk.Button(
                self.palette_frame,
                bg=hx,
                width=4, height=2,
                relief='raised',
                cursor='hand2'
            )
            btn.grid(row=idx//10, column=idx%10, padx=pad, pady=pad)
            btn.bind("<Button-3>", lambda ev, i=idx: self.delete_color(i))

        plus_btn = ctk.CTkButton(
            self.palette_frame,
            text="+",
            width=square_size,
            height=square_size,
            command=self.add_color,
            corner_radius=8
        )
        plus_btn.grid(row=len(self.colors)//10, column=len(self.colors)%10, padx=pad, pady=pad)

    def add_color(self):
        pick = HSVColorPickerDialog(self)
        self.wait_window(pick)
        if pick.selected_color is not None:
            self.colors.append(pick.selected_color)
            self.update_palette_display()

    def delete_color(self, index: int):
        if 0 <= index < len(self.colors):
            self.colors.pop(index)
            self.update_palette_display()

    def save_palette(self):
        if not self.colors:
            messagebox.showwarning("No Colors", "Please add at least one color to the palette.", parent=self)
            return
        pname = simpledialog.askstring("Palette Name", "Enter a name for the custom palette:", parent=self)
        if not pname:
            return
        
        # Convert RGB tuples to hex strings
        hex_colors = [f'#{col[0]:02x}{col[1]:02x}{col[2]:02x}' for col in self.colors]
        
        # Add palette (will update if exists)
        self.palette_manager.add_palette(pname, hex_colors)
        self.refresh_callback()
        self.destroy()


# -------------------- Palette Image Preview Dialog --------------------

class PaletteImagePreviewDialog(ctk.CTkToplevel):
    """
    A small dialog to confirm or re-try a generated palette from an image.
    """
    def __init__(self, parent, palette, file_path, used_clusters):
        super().__init__(parent)
        self.title("New Palette Preview")
        self.geometry("400x180")
        self.resizable(False, False)

        self.use_result = False
        self.choose_another = False

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        bn = os.path.basename(file_path)
        info = ctk.CTkLabel(
            self,
            text=f"Generated a {used_clusters}-color palette from:\n{bn}\n\nUse this palette or pick another image?"
        )
        info.pack(pady=(10, 0))

        self.preview = PalettePreview(self, palette, width=300, height=30)
        self.preview.pack(pady=10)

        bf = ctk.CTkFrame(self)
        bf.pack(pady=5, fill='x')

        ub = ctk.CTkButton(bf, text="Use This Palette", command=self.use_palette)
        ub.pack(side='left', expand=True, fill='x', padx=5, pady=5)
        ab = ctk.CTkButton(bf, text="Choose Another Image", command=self.pick_another)
        ab.pack(side='right', expand=True, fill='x', padx=5, pady=5)

    def use_palette(self):
        self.use_result = True
        self.destroy()

    def pick_another(self):
        self.choose_another = True
        self.destroy()


# -------------------- Dither Mode Settings Dialog --------------------

class DitherSettingsDialog(ctk.CTkToplevel):
    """
    Generic settings dialog for dithering mode parameters.
    Automatically builds UI based on parameter metadata.
    """
    def __init__(self, parent, mode_name: str, parameter_info: dict,
                 current_values: dict = None, on_change=None, on_cancel=None, on_apply=None):
        super().__init__(parent)
        self.title(f"{mode_name} Settings")
        self.geometry("450x600")
        self.resizable(False, True)
        
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()
        
        self.parameter_info = parameter_info
        self.current_values = current_values or {}
        self.result_values = None
        self.widgets = {}
        self.on_change = on_change
        self.on_cancel_callback = on_cancel
        self.on_apply_callback = on_apply
        self._live_change_job = None
        
        # Title
        title = ctk.CTkLabel(
            self,
            text=f"Configure {mode_name}",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(15, 10))
        
        # Scrollable frame for parameters
        self.scroll_frame = ctk.CTkScrollableFrame(self, height=400)
        self.scroll_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Build parameter controls
        self._build_parameter_widgets()
        
        # Buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=15, padx=15, fill='x')
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.on_cancel,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Apply",
            command=self.on_apply,
            width=120
        ).pack(side="right", padx=5)
        
        # Reset button in the middle
        ctk.CTkButton(
            btn_frame,
            text="Reset to Defaults",
            command=self.on_reset,
            width=140
        ).pack(side="left", padx=20, expand=True)
    
    def _build_parameter_widgets(self):
        """Build UI widgets for each parameter."""
        row = 0
        
        for param_name, param_info in self.parameter_info.items():
            # Label
            label_text = param_info.get('label', param_name)
            label = ctk.CTkLabel(
                self.scroll_frame,
                text=label_text + ":",
                font=("Arial", 12, "bold")
            )
            label.grid(row=row, column=0, sticky='w', padx=10, pady=(10, 2))
            row += 1
            
            # Description
            desc = param_info.get('description', '')
            if desc:
                desc_label = ctk.CTkLabel(
                    self.scroll_frame,
                    text=desc,
                    font=("Arial", 10),
                    text_color="gray"
                )
                desc_label.grid(row=row, column=0, sticky='w', padx=10, pady=(0, 5))
                row += 1
            
            # Widget based on type
            param_type = param_info.get('type')
            current_value = self.current_values.get(param_name, param_info.get('default'))
            
            if param_type == 'int':
                widget = self._create_int_widget(param_name, param_info, current_value)
            elif param_type == 'float':
                widget = self._create_float_widget(param_name, param_info, current_value)
            elif param_type == 'choice':
                widget = self._create_choice_widget(param_name, param_info, current_value)
            else:
                widget = None
            
            if widget:
                widget.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 5))
                self.widgets[param_name] = widget
                row += 1
        
        # Configure column weight
        self.scroll_frame.grid_columnconfigure(0, weight=1)
    
    def _create_int_widget(self, param_name, param_info, current_value):
        """Create widget for integer parameter."""
        frame = ctk.CTkFrame(self.scroll_frame)
        
        min_val = param_info.get('min', 0)
        max_val = param_info.get('max', 100)
        
        # Entry field
        entry = ctk.CTkEntry(frame, width=80)
        entry.insert(0, str(int(current_value)))
        entry.bind("<KeyRelease>", lambda _e: self._schedule_live_change())
        entry.bind("<FocusOut>", lambda _e: self._schedule_live_change())
        entry.pack(side='left', padx=(0, 10))
        
        # Range label
        range_label = ctk.CTkLabel(
            frame,
            text=f"({min_val} - {max_val})",
            font=("Arial", 10),
            text_color="gray"
        )
        range_label.pack(side='left')
        
        frame.entry = entry
        frame.min_val = min_val
        frame.max_val = max_val
        
        return frame
    
    def _create_float_widget(self, param_name, param_info, current_value):
        """Create widget for float parameter."""
        frame = ctk.CTkFrame(self.scroll_frame)
        
        min_val = param_info.get('min', 0.0)
        max_val = param_info.get('max', 1.0)
        step = param_info.get('step', 0.1)
        
        # Entry field
        entry = ctk.CTkEntry(frame, width=80)
        entry.insert(0, f"{current_value:.2f}")
        entry.bind("<KeyRelease>", lambda _e: self._schedule_live_change())
        entry.bind("<FocusOut>", lambda _e: self._schedule_live_change())
        entry.pack(side='left', padx=(0, 10))
        
        # Range label
        range_label = ctk.CTkLabel(
            frame,
            text=f"({min_val} - {max_val}, step {step})",
            font=("Arial", 10),
            text_color="gray"
        )
        range_label.pack(side='left')
        
        frame.entry = entry
        frame.min_val = min_val
        frame.max_val = max_val
        frame.step = step
        
        return frame
    
    def _create_choice_widget(self, param_name, param_info, current_value):
        """Create widget for choice parameter."""
        choices = param_info.get('choices', [])
        
        var = tk.StringVar(value=current_value)
        menu = ctk.CTkOptionMenu(
            self.scroll_frame,
            variable=var,
            values=choices,
            command=lambda _v: self._schedule_live_change()
        )
        menu.var = var
        
        return menu
    
    def _get_widget_value(self, param_name):
        """Extract value from widget with validation."""
        widget = self.widgets.get(param_name)
        if not widget:
            return None
        
        param_info = self.parameter_info[param_name]
        param_type = param_info.get('type')
        
        if param_type == 'int':
            try:
                value = int(widget.entry.get())
                # Clamp to valid range
                value = max(widget.min_val, min(widget.max_val, value))
                return value
            except ValueError:
                # Return default if invalid
                return param_info.get('default', widget.min_val)
        elif param_type == 'float':
            try:
                value = float(widget.entry.get())
                # Clamp to valid range
                value = max(widget.min_val, min(widget.max_val, value))
                return value
            except ValueError:
                # Return default if invalid
                return param_info.get('default', widget.min_val)
        elif param_type == 'choice':
            return widget.var.get()
        
        return None
    
    def on_reset(self):
        """Reset all parameters to defaults."""
        for param_name, widget in self.widgets.items():
            param_info = self.parameter_info[param_name]
            default_value = param_info.get('default')
            param_type = param_info.get('type')
            
            if param_type == 'int':
                widget.entry.delete(0, tk.END)
                widget.entry.insert(0, str(int(default_value)))
            elif param_type == 'float':
                widget.entry.delete(0, tk.END)
                widget.entry.insert(0, f"{default_value:.2f}")
            elif param_type == 'choice':
                widget.var.set(default_value)
        self._schedule_live_change()

    def _schedule_live_change(self):
        """Debounce live change notifications."""
        if not self.on_change:
            return
        if self._live_change_job is not None:
            self.after_cancel(self._live_change_job)
        self._live_change_job = self.after(250, self._emit_live_change)

    def _emit_live_change(self):
        """Emit current parameter values for live preview."""
        self._live_change_job = None
        if not self.on_change:
            return
        current_values = {}
        for param_name in self.parameter_info.keys():
            current_values[param_name] = self._get_widget_value(param_name)
        self.on_change(current_values)
    
    def on_apply(self):
        """Apply settings and close."""
        self.result_values = {}
        for param_name in self.parameter_info.keys():
            self.result_values[param_name] = self._get_widget_value(param_name)
        if self.on_apply_callback:
            self.on_apply_callback(self.result_values)
        self.destroy()
    
    def on_cancel(self):
        """Cancel and close."""
        self.result_values = None
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.destroy()


class PixelizationEditorDialog(ctk.CTkToplevel):
    """
    Dialog for pixelization setup, conversion, and editing.
    """
    def __init__(self, parent, source_image: Image.Image, config=None):
        super().__init__(parent)
        self.title("Pixelization Editor")
        self.resizable(True, True)
        self.minsize(900, 600)
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()
        try:
            self.attributes("-toolwindow", 0)
        except tk.TclError:
            pass
        self.after(0, self._enable_maximize_box)

        self.config_mgr = config
        self.source_image = source_image.convert("RGBA")
        self.result_image = None
        self.grid_w = 1
        self.grid_h = 1
        self._alt_pick_active = False
        self._alt_prev_tool = None

        self._build_ui()
        self._load_geometry()
        self._apply_target_size()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, width=240)
        left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        left.grid_propagate(False)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.canvas = PixelizationEditorCanvas(
            right, bg="gray20", highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.set_source_image(self.source_image)
        self.canvas.set_image(self.source_image, update=False)
        self.canvas.fit_to_window()
        self.canvas.set_mode("preview")
        self.canvas.set_show_grid(True)

        # Setup section
        self._section_label(left, "Setup")
        target_frame = ctk.CTkFrame(left, fg_color="transparent")
        target_frame.pack(fill="x", padx=10, pady=(2, 6))
        ctk.CTkLabel(target_frame, text="Target size:").pack(side="left")
        self.target_size_entry = ctk.CTkEntry(target_frame, width=80)
        default_size = 64
        if self.config_mgr:
            default_size = self.config_mgr.get("pixelization_editor", "target_size", default=64)
        self.target_size_entry.insert(0, str(default_size))
        self.target_size_entry.pack(side="left", padx=6)
        self.target_size_entry.bind("<Return>", lambda _e: self._apply_target_size())
        self.target_size_entry.bind("<FocusOut>", lambda _e: self._apply_target_size())

        grid_row = ctk.CTkFrame(left, fg_color="transparent")
        grid_row.pack(fill="x", padx=10, pady=(0, 6))
        self.grid_label = ctk.CTkLabel(grid_row, text="Grid: -")
        self.grid_label.pack(side="left")
        self.fit_to_grid_button = ctk.CTkButton(
            grid_row,
            text="Fit to grid",
            width=90,
            command=self.canvas.fit_image_to_grid
        )
        self.fit_to_grid_button.pack(side="left", padx=8)

        # Convert section
        self._section_label(left, "Convert")
        self.convert_button = ctk.CTkButton(
            left,
            text="Pixelize",
            command=self._start_conversion
        )
        self.convert_button.pack(fill="x", padx=10, pady=(4, 10))

        # Edit section
        self._section_label(left, "Edit")
        tool_frame = ctk.CTkFrame(left, fg_color="transparent")
        tool_frame.pack(fill="x", padx=10, pady=(2, 6))

        self.tool_buttons = {}
        self.tool_buttons["brush"] = ctk.CTkButton(
            tool_frame, text="Brush", width=70, height=28,
            command=lambda: self._set_tool("brush")
        )
        self.tool_buttons["brush"].pack(side="left", padx=2)
        self.tool_buttons["magic"] = ctk.CTkButton(
            tool_frame, text="Wand", width=70, height=28,
            command=lambda: self._set_tool("magic")
        )
        self.tool_buttons["magic"].pack(side="left", padx=2)
        self.tool_buttons["picker"] = ctk.CTkButton(
            tool_frame, text="Pick", width=70, height=28,
            command=lambda: self._set_tool("picker")
        )
        self.tool_buttons["picker"].pack(side="left", padx=2)
        self._tool_default_fg = {
            name: btn.cget("fg_color")
            for name, btn in self.tool_buttons.items()
        }

        size_frame = ctk.CTkFrame(left, fg_color="transparent")
        size_frame.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkLabel(size_frame, text="Tool size:").pack(side="left")
        self.tool_size_var = tk.StringVar(
            value=str(self._get_config_value("tool_size", 1))
        )
        self.tool_size_entry = tk.Spinbox(
            size_frame,
            from_=1,
            to=9999,
            width=6,
            textvariable=self.tool_size_var,
            command=self._apply_tool_size
        )
        self.tool_size_entry.pack(side="left", padx=6)
        self.tool_size_entry.bind("<KeyRelease>", lambda _e: self._apply_tool_size())

        color_frame = ctk.CTkFrame(left, fg_color="transparent")
        color_frame.pack(fill="x", padx=10, pady=(0, 6))
        ctk.CTkLabel(color_frame, text="Color:").pack(side="left")
        self.color_button = ctk.CTkButton(
            color_frame,
            text="",
            width=28,
            height=28,
            command=self._pick_color
        )
        self.color_button.pack(side="left", padx=6)
        self._set_draw_color_from_hex("#000000")

        self.grid_toggle_var = tk.BooleanVar(
            value=self._get_config_value("grid_visible", True)
        )
        self.grid_toggle = ctk.CTkCheckBox(
            left,
            text="Show grid",
            variable=self.grid_toggle_var,
            command=self._apply_grid_toggle
        )
        self.grid_toggle.pack(anchor="w", padx=10, pady=(0, 6))

        self.magic_threshold = tk.IntVar(
            value=self._get_config_value("magic_wand_threshold", 5)
        )
        self.magic_label = ctk.CTkLabel(left, text=f"Wand threshold: {self.magic_threshold.get()}")
        self.magic_label.pack(anchor="w", padx=10, pady=(0, 2))
        self._magic_label_color = self.magic_label.cget("text_color")
        self.magic_slider = ctk.CTkSlider(
            left,
            from_=0,
            to=30,
            number_of_steps=30,
            command=self._apply_magic_threshold
        )
        self.magic_slider.set(self.magic_threshold.get())
        self.magic_slider.pack(fill="x", padx=10, pady=(0, 6))

        history_frame = ctk.CTkFrame(left, fg_color="transparent")
        history_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.undo_button = ctk.CTkButton(
            history_frame, text="Undo", width=70, command=self._undo
        )
        self.undo_button.pack(side="left", padx=2)
        self.redo_button = ctk.CTkButton(
            history_frame, text="Redo", width=70, command=self._redo
        )
        self.redo_button.pack(side="left", padx=2)

        # Bottom buttons
        bottom = ctk.CTkFrame(left, fg_color="transparent")
        bottom.pack(fill="x", padx=10, pady=(10, 0))
        self.cancel_button = ctk.CTkButton(bottom, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side="left", expand=True, fill="x", padx=2)
        self.apply_button = ctk.CTkButton(bottom, text="Apply", command=self._on_apply)
        self.apply_button.pack(side="left", expand=True, fill="x", padx=2)

        self._set_edit_controls_state(False)
        tool = self._get_config_value("tool", "brush")
        if tool not in self.tool_buttons:
            tool = "brush"
        self._set_tool(tool)
        self._bind_shortcuts()

    def _section_label(self, parent, text: str):
        label = ctk.CTkLabel(parent, text=text, font=("Arial", 12, "bold"))
        label.pack(anchor="w", padx=10, pady=(8, 2))

    def _get_config_value(self, key: str, default):
        if not self.config_mgr:
            return default
        return self.config_mgr.get("pixelization_editor", key, default=default)

    def _load_geometry(self):
        if not self.config_mgr:
            self.geometry("1100x700")
            return
        w = self.config_mgr.get("pixelization_editor", "dialog_width", default=1100)
        h = self.config_mgr.get("pixelization_editor", "dialog_height", default=700)
        x = self.config_mgr.get("pixelization_editor", "dialog_x")
        y = self.config_mgr.get("pixelization_editor", "dialog_y")
        if x is not None and y is not None:
            self.geometry(f"{w}x{h}+{x}+{y}")
        else:
            self.geometry(f"{w}x{h}")

    def _enable_maximize_box(self):
        if os.name != "nt":
            return
        try:
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if not hwnd:
                return
            GWL_STYLE = -16
            WS_MAXIMIZEBOX = 0x00010000
            WS_THICKFRAME = 0x00040000
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            style |= WS_MAXIMIZEBOX | WS_THICKFRAME
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
            ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0027)
        except Exception:
            return

    def _save_geometry(self):
        if not self.config_mgr:
            return
        geom = self.geometry()
        size_pos = geom.split("+")
        size = size_pos[0].split("x")
        self.config_mgr.set("pixelization_editor", "dialog_width", value=int(size[0]))
        self.config_mgr.set("pixelization_editor", "dialog_height", value=int(size[1]))
        if len(size_pos) >= 3:
            self.config_mgr.set("pixelization_editor", "dialog_x", value=int(size_pos[1]))
            self.config_mgr.set("pixelization_editor", "dialog_y", value=int(size_pos[2]))

    def _apply_target_size(self):
        try:
            target = int(self.target_size_entry.get())
            if target <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Size", "Target size must be a positive integer.")
            return
        img_w, img_h = self.source_image.size
        if img_w >= img_h:
            grid_h = target
            grid_w = max(1, int(round((img_w / img_h) * target)))
        else:
            grid_w = target
            grid_h = max(1, int(round((img_h / img_w) * target)))
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.canvas.set_grid(grid_w, grid_h)
        self.grid_label.configure(text=f"Grid: {grid_w} x {grid_h}")
        self._set_edit_controls_state(False)
        self.canvas.set_mode("preview")
        self.canvas.set_show_grid(True)
        self.canvas.set_image(self.source_image, update=False)
        self.canvas.update_idletasks()
        self.canvas.fit_image_to_grid()

    def _set_tool(self, tool: str):
        self.canvas.set_tool(tool)
        for name, btn in self.tool_buttons.items():
            if name == tool:
                btn.configure(fg_color="#2d7ff9")
            else:
                btn.configure(fg_color=self._tool_default_fg.get(name))
        if tool == "magic":
            self.magic_slider.configure(state="normal")
            self.magic_label.configure(text_color=self._magic_label_color)
        else:
            self.magic_slider.configure(state="disabled")
            self.magic_label.configure(text_color="gray")
        if self.config_mgr:
            self.config_mgr.set("pixelization_editor", "tool", value=tool)

    def _apply_tool_size(self):
        try:
            size = int(self.tool_size_var.get())
        except ValueError:
            size = 1
        if size <= 0:
            size = 1
        self.canvas.set_tool_size(size)
        if self.config_mgr:
            self.config_mgr.set("pixelization_editor", "tool_size", value=size)

    def _pick_color(self):
        current = self.color_button.cget("fg_color")
        rgb, hex_color = colorchooser.askcolor(color=current, parent=self)
        if hex_color:
            self._set_draw_color_from_hex(hex_color)

    def _set_draw_color_from_hex(self, hex_color: str):
        hex_color = hex_color.lower()
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        if len(hex_color) != 6:
            return
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        self.canvas.set_draw_color((r, g, b))
        self.color_button.configure(fg_color=f"#{hex_color}")

    def _apply_grid_toggle(self):
        self.canvas.set_show_grid(self.grid_toggle_var.get())
        if self.config_mgr:
            self.config_mgr.set("pixelization_editor", "grid_visible", value=self.grid_toggle_var.get())

    def _apply_magic_threshold(self, value):
        self.magic_threshold.set(int(value))
        self.canvas.set_magic_threshold(self.magic_threshold.get())
        self.magic_label.configure(text=f"Wand threshold: {self.magic_threshold.get()}")
        if self.config_mgr:
            self.config_mgr.set(
                "pixelization_editor",
                "magic_wand_threshold",
                value=self.magic_threshold.get()
            )

    def _set_edit_controls_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        widgets = [
            self.tool_buttons["brush"],
            self.tool_buttons["magic"],
            self.tool_buttons["picker"],
            self.tool_size_entry,
            self.color_button,
            self.grid_toggle,
            self.magic_slider,
            self.undo_button,
            self.redo_button,
            self.apply_button
        ]
        for w in widgets:
            w.configure(state=state)
        if enabled:
            self._set_tool(self.canvas.tool)

    def _start_conversion(self):
        self.convert_button.configure(state="disabled", text="Pixelizing...")
        threading.Thread(target=self._convert, daemon=True).start()

    def _convert(self):
        rect = self.canvas._get_image_draw_rect()
        if not rect:
            self.after(0, lambda: self._conversion_done(None, "No image to convert."))
            return
        transform = self.canvas._get_image_transform()
        if not transform:
            self.after(0, lambda: self._conversion_done(None, "No image to convert."))
            return
        grid_rect = self.canvas._get_preview_grid_rect()
        if not grid_rect:
            self.after(0, lambda: self._conversion_done(None, "Grid not ready yet."))
            return
        origin_x, origin_y, scale = transform
        grid_x, grid_y, _, _, cell = grid_rect
        img_arr = np.array(self.source_image)
        img_h, img_w = img_arr.shape[0], img_arr.shape[1]
        has_alpha = img_arr.shape[2] == 4
        pixels = [[None for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        for j in range(self.grid_h):
            for i in range(self.grid_w):
                cx = grid_x + (i + 0.5) * cell
                cy = grid_y + (j + 0.5) * cell
                ox = int(round((cx - origin_x) / scale))
                oy = int(round((cy - origin_y) / scale))
                if ox < 0 or oy < 0 or ox >= img_w or oy >= img_h:
                    continue
                if has_alpha and img_arr[oy, ox, 3] == 0:
                    continue
                r, g, b = img_arr[oy, ox, 0:3]
                pixels[j][i] = (int(r), int(g), int(b))
        self.after(0, lambda: self._conversion_done(pixels, None))

    def _conversion_done(self, pixels, _message):
        if pixels is None:
            self.convert_button.configure(state="normal", text="Pixelize")
            return
        self.canvas.set_pixel_data(pixels)
        self.canvas.set_mode("edit")
        self.canvas.set_show_grid(self.grid_toggle_var.get())
        self._apply_tool_size()
        self.canvas.set_magic_threshold(self.magic_threshold.get())
        self.canvas.on_color_pick = self._on_color_picked
        self._set_edit_controls_state(True)
        self.convert_button.configure(state="normal", text="Pixelize")
        if self.config_mgr:
            self.config_mgr.set("pixelization_editor", "conversion_method", value="neighbor")


    def _undo(self):
        self.canvas.undo()

    def _redo(self):
        self.canvas.redo_action()

    def _on_color_picked(self, color: Tuple[int, int, int]):
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        self.color_button.configure(fg_color=hex_color)

    def _bind_shortcuts(self):
        def _undo(_e=None):
            self._undo()
            return "break"

        def _redo(_e=None):
            self._redo()
            return "break"

        self.bind("<Control-z>", _undo)
        self.bind("<Control-y>", _redo)
        self.bind("<Control-Shift-Z>", _redo)
        self.bind_all("<KeyPress-Alt_L>", self._on_alt_down)
        self.bind_all("<KeyPress-Alt_R>", self._on_alt_down)
        self.bind_all("<KeyRelease-Alt_L>", self._on_alt_up)
        self.bind_all("<KeyRelease-Alt_R>", self._on_alt_up)

    def _is_active_dialog(self) -> bool:
        widget = self.focus_get()
        if not widget:
            return False
        return widget.winfo_toplevel() == self

    def _on_alt_down(self, _event):
        if not self._is_active_dialog():
            return
        if self._alt_pick_active:
            return
        self.canvas.alt_zoom_active = True
        if self.canvas.tool == "picker":
            return
        self._alt_prev_tool = self.canvas.tool
        self._alt_pick_active = True
        self._set_tool("picker")

    def _on_alt_up(self, _event):
        self.canvas.alt_zoom_active = False
        if not self._alt_pick_active:
            return
        self._alt_pick_active = False
        prev_tool = self._alt_prev_tool or "brush"
        self._alt_prev_tool = None
        self._set_tool(prev_tool)

    def _on_apply(self):
        if not self.canvas.get_pixel_data():
            messagebox.showwarning("No Result", "Convert an image before applying.")
            return
        self.result_image = self._pixel_data_to_image(self.canvas.get_pixel_data())
        self._save_config()
        self._save_geometry()
        self.destroy()

    def _pixel_data_to_image(self, pixels):
        img = Image.new("RGB", (self.grid_w, self.grid_h), (0, 0, 0))
        px = img.load()
        for j in range(self.grid_h):
            for i in range(self.grid_w):
                color = pixels[j][i]
                if color is None:
                    continue
                px[i, j] = color
        return img

    def _save_config(self):
        if not self.config_mgr:
            return
        self.config_mgr.set("pixelization_editor", "target_size", value=int(self.target_size_entry.get()))
        self.config_mgr.set("pixelization_editor", "conversion_method", value="neighbor")
        self.config_mgr.set("pixelization_editor", "tool_size", value=int(self.tool_size_var.get()))
        self.config_mgr.set("pixelization_editor", "grid_visible", value=self.grid_toggle_var.get())
        self.config_mgr.set(
            "pixelization_editor",
            "magic_wand_threshold",
            value=self.magic_threshold.get()
        )

    def _on_cancel(self):
        self.result_image = None
        self._save_config()
        self._save_geometry()
        self.destroy()