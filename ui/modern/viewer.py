"""Canvas viewer: zoom, pan, transparency checkerboard, before/after split."""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Optional

from PIL import Image, ImageTk

from . import theme as T


class ImageCanvas(tk.Canvas):
    """Shows one image, or two split by a draggable handle.

    Only the visible part of the source is ever resampled, so zooming deep
    into a large image stays cheap.
    """

    def __init__(self, master, on_zoom_change: Callable = None, **kw):
        super().__init__(master, highlightthickness=0, bd=0,
                         bg=T.res(T.c("canvas")), **kw)
        self.image_a: Optional[Image.Image] = None
        self.image_b: Optional[Image.Image] = None
        self.compare = False
        self.split = 0.5
        self.show_pixel_grid = True

        self.zoom = 1.0
        self.off_x = 0.0
        self.off_y = 0.0

        self._photo = None
        self._drag = None
        self._split_drag = False
        self._on_zoom_change = on_zoom_change
        self._checker_cache = None
        self._redraw_id = None
        self._aligned_compare = None

        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<Motion>", self._hover)

    # -- public ----------------------------------------------------------

    def set_image(self, img: Optional[Image.Image], fit=True):
        previous = self.image_a
        if not fit and previous is not None and img is not None:
            # Keep the same scene footprint when alternating resolutions.
            self.zoom *= previous.width / img.width
        self.image_a = img
        if fit and img is not None:
            self.fit()
        else:
            self.redraw()
            self._emit_zoom()

    def set_compare_image(self, img: Optional[Image.Image]):
        self.image_b = img
        self._aligned_compare = None
        self.redraw()

    def set_compare(self, on: bool):
        self.compare = on and self.image_b is not None
        self.redraw()

    def fit(self):
        img = self.image_a
        if img is None:
            return
        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        if vw < 60 or vh < 60:
            # Called before the first <Configure>; retry once laid out.
            self.after(60, self.fit)
            return
        pad = 48
        z = min((vw - pad) / img.width, (vh - pad) / img.height)
        self.zoom = max(0.02, min(z, 24.0))
        self.off_x = self.off_y = 0.0
        self.redraw()
        self._emit_zoom()

    def actual_size(self):
        if self.image_a is None:
            return
        self.zoom = 1.0
        self.off_x = self.off_y = 0.0
        self.redraw()
        self._emit_zoom()

    def zoom_by(self, factor: float):
        if self.image_a is None:
            return
        self.zoom = max(0.02, min(64.0, self.zoom * factor))
        self.redraw()
        self._emit_zoom()

    # -- painting --------------------------------------------------------

    def _emit_zoom(self):
        if self._on_zoom_change:
            self._on_zoom_change(self.zoom)

    def _checker(self, w, h):
        """Transparency backdrop, cached per size and appearance mode."""
        key = (w, h, T.is_dark())
        if self._checker_cache and self._checker_cache[0] == key:
            return self._checker_cache[1]
        cell = 8
        light = (52, 57, 66) if T.is_dark() else (222, 226, 232)
        dark = (40, 44, 52) if T.is_dark() else (200, 205, 213)
        tile = Image.new("RGB", (cell * 2, cell * 2), light)
        for x, y in ((0, 0), (cell, cell)):
            tile.paste(dark, (x, y, x + cell, y + cell))
        # Grow by doubling rather than thousands of Python-level tile pastes.
        bg = tile
        while bg.width < w or bg.height < h:
            grown = Image.new('RGB', (min(w, bg.width * 2), min(h, bg.height * 2)))
            for y in (0, bg.height):
                for x in (0, bg.width):
                    grown.paste(bg, (x, y))
            bg = grown
        bg = bg.crop((0, 0, w, h))
        self._checker_cache = (key, bg)
        return bg

    def geometry_of_image(self):
        """Top-left of the drawn image in canvas coordinates, plus its size."""
        img = self.image_a
        if img is None:
            return None
        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        dw, dh = img.width * self.zoom, img.height * self.zoom
        ox = (vw - dw) / 2 + self.off_x
        oy = (vh - dh) / 2 + self.off_y
        return ox, oy, dw, dh

    def redraw(self):
        # Coalesce motion/configure bursts to one frame per ~16 ms.
        if self._redraw_id is None:
            self._redraw_id = self.after(16, self._render)

    def destroy(self):
        if self._redraw_id is not None:
            self.after_cancel(self._redraw_id)
            self._redraw_id = None
        super().destroy()

    def _render(self):
        self._redraw_id = None
        self.delete("all")
        self.configure(bg=T.res(T.c("canvas")))
        img = self.image_a
        if img is None:
            self._draw_empty()
            return

        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        ox, oy, dw, dh = self.geometry_of_image()

        # Visible window in source pixels.
        sx0 = max(0.0, (0 - ox) / self.zoom)
        sy0 = max(0.0, (0 - oy) / self.zoom)
        sx1 = min(float(img.width), (vw - ox) / self.zoom)
        sy1 = min(float(img.height), (vh - oy) / self.zoom)
        if sx1 <= sx0 or sy1 <= sy0:
            return

        cx0, cy0 = int(math.floor(sx0)), int(math.floor(sy0))
        cx1, cy1 = int(math.ceil(sx1)), int(math.ceil(sy1))
        tw = max(1, int(round((cx1 - cx0) * self.zoom)))
        th = max(1, int(round((cy1 - cy0) * self.zoom)))
        resample = Image.NEAREST if self.zoom >= 1.0 else Image.LANCZOS

        tile = img.crop((cx0, cy0, cx1, cy1)).resize((tw, th), resample)

        if self.compare and self.image_b is not None:
            b = self.image_b
            # Sample each original independently into the same screen rect.
            # Downsampling the original to the processed resolution first
            # would erase exactly the detail the comparison should reveal.
            box = (cx0 * b.width / img.width, cy0 * b.height / img.height,
                   cx1 * b.width / img.width, cy1 * b.height / img.height)
            b_filter = Image.NEAREST if tw >= box[2] - box[0] else Image.LANCZOS
            tile_b = b.resize((tw, th), b_filter, box=box)
            split_x = int(vw * self.split)
            local = int(round(split_x - (ox + cx0 * self.zoom)))
            local = max(0, min(tw, local))
            # Left of the handle is the "before" image, right is the result.
            if local > 0:
                tile.paste(tile_b.crop((0, 0, local, th)), (0, 0))

        if tile.mode in ("RGBA", "LA", "P"):
            tile = tile.convert("RGBA")
            bg = self._checker(tw, th).copy()
            bg.paste(tile, (0, 0), tile)
            tile = bg

        tx = ox + cx0 * self.zoom
        ty = oy + cy0 * self.zoom
        self._photo = ImageTk.PhotoImage(tile)
        self.create_image(tx, ty, image=self._photo, anchor="nw")

        self._draw_border(ox, oy, dw, dh)
        if self.show_pixel_grid and self.zoom >= 10:
            self._draw_pixel_grid(ox, oy, cx0, cy0, cx1, cy1)
        if self.compare and self.image_b is not None:
            self._draw_split(vw, vh)

    def _draw_border(self, ox, oy, dw, dh):
        col = T.mix(T.res(T.c("canvas")), "#FFFFFF" if T.is_dark() else "#000000", 0.16)
        self.create_rectangle(ox - 1, oy - 1, ox + dw, oy + dh,
                              outline=col, width=1)

    def _draw_pixel_grid(self, ox, oy, cx0, cy0, cx1, cy1):
        col = T.mix(T.res(T.c("canvas")), "#FFFFFF" if T.is_dark() else "#000000", 0.22)
        for i in range(cx0, cx1 + 1):
            x = ox + i * self.zoom
            self.create_line(x, oy + cy0 * self.zoom, x, oy + cy1 * self.zoom,
                             fill=col)
        for j in range(cy0, cy1 + 1):
            y = oy + j * self.zoom
            self.create_line(ox + cx0 * self.zoom, y, ox + cx1 * self.zoom, y,
                             fill=col)

    def _draw_split(self, vw, vh):
        ox, oy, dw, dh = self.geometry_of_image()
        x = vw * self.split
        # Keep the divider inside the image so it reads as a wipe over the
        # picture rather than a line across the whole workspace.
        top = max(0, min(vh, oy))
        bottom = max(0, min(vh, oy + dh))
        acc = T.res(T.accent())
        self.create_line(x, top, x, bottom, fill=acc, width=2, tags="split")
        cy = (top + bottom) / 2
        self.create_oval(x - 15, cy - 15, x + 15, cy + 15, fill=acc,
                         outline="", tags="split")
        self.create_text(x, cy, text="↔", fill=T.res(T.accent_on()),
                         font=(T.FONTS["_raw_ui"], 12, "bold"), tags="split")
        label_y = min(vh - 14, bottom + 10)
        self.create_text(max(10, ox), label_y, anchor="nw", text="BEFORE",
                         fill=T.res(T.c("faint")),
                         font=(T.FONTS["_raw_ui"], 9, "bold"))
        self.create_text(min(vw - 10, ox + dw), label_y, anchor="ne",
                         text="AFTER", fill=T.res(T.c("faint")),
                         font=(T.FONTS["_raw_ui"], 9, "bold"))

    def _draw_empty(self):
        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        cx, cy = vw / 2, vh / 2
        col = T.res(T.c("faint"))
        box = 54
        T.round_rect(self, cx - box, cy - box - 6, cx + box, cy + box - 6, 18,
                     fill="", outline=col, width=2, dash=(6, 6))
        self.create_text(cx, cy - 18, text=T.ICON["image"],
                         fill=col, font=("Segoe Fluent Icons", 30))
        self.create_text(cx, cy + 22, text="Open an image or video",
                         fill=T.res(T.c("dim")),
                         font=(T.FONTS["_raw_ui"], 12, "bold"))
        self.create_text(cx, cy + 44, text="Use the Source panel or press Ctrl+O",
                         fill=col, font=(T.FONTS["_raw_ui"], 10))

    # -- interaction -----------------------------------------------------

    def _near_split(self, x, y) -> bool:
        if not (self.compare and self.image_b is not None):
            return False
        return abs(x - self.winfo_width() * self.split) <= 14

    def _hover(self, e):
        if self._near_split(e.x, e.y):
            self.configure(cursor="sb_h_double_arrow")
        elif self.image_a is not None:
            self.configure(cursor="fleur")
        else:
            self.configure(cursor="")

    def _press(self, e):
        if self._near_split(e.x, e.y):
            self._split_drag = True
            return
        self._drag = (e.x, e.y, self.off_x, self.off_y)

    def _motion(self, e):
        if self._split_drag:
            self.split = max(0.02, min(0.98, e.x / max(1, self.winfo_width())))
            self.redraw()
            return
        if self._drag and self.image_a is not None:
            x0, y0, ox, oy = self._drag
            self.off_x = ox + (e.x - x0)
            self.off_y = oy + (e.y - y0)
            self.redraw()

    def _release(self, _e):
        self._drag = None
        self._split_drag = False

    def _wheel(self, e):
        if self.image_a is None:
            return
        # Keep the point under the cursor anchored while zooming.
        ox, oy, _, _ = self.geometry_of_image()
        sx = (e.x - ox) / self.zoom
        sy = (e.y - oy) / self.zoom
        factor = 1.15 if e.delta > 0 else 1 / 1.15
        self.zoom = max(0.02, min(64.0, self.zoom * factor))
        vw, vh = max(1, self.winfo_width()), max(1, self.winfo_height())
        dw, dh = self.image_a.width * self.zoom, self.image_a.height * self.zoom
        self.off_x = e.x - sx * self.zoom - (vw - dw) / 2
        self.off_y = e.y - sy * self.zoom - (vh - dh) / 2
        self.redraw()
        self._emit_zoom()
