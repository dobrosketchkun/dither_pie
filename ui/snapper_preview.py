"""Two preview canvases sharing one camera in source-image coordinates."""
import math
import tkinter as tk

from PIL import Image, ImageTk

from .image_view import ZoomableImage


class PreviewCamera:
    def __init__(self, scene_size):
        self.scene_size = scene_size
        self.views = []
        self.scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self.auto_fit = True

    def redraw(self):
        for view in self.views:
            view.update_view()

    def fit(self):
        if not self.views or any(min(v.winfo_width(), v.winfo_height()) < 2 for v in self.views):
            return
        width = min(v.winfo_width() for v in self.views)
        height = min(v.winfo_height() for v in self.views)
        self.scale = min(width / self.scene_size[0], height / self.scene_size[1])
        self.offset_x = self.offset_y = 0.0
        self.auto_fit = True
        self.redraw()


class SyncedPreview(tk.Canvas):
    def __init__(self, master, camera, **kwargs):
        super().__init__(master, **kwargs)
        self.camera = camera
        camera.views.append(self)
        self.original_image = None
        self._photo = None
        self._pan_start = None
        self.pick_callback = None
        self.bind('<Configure>', self.on_resize)
        self.bind('<ButtonPress-1>', self.start_pan)
        self.bind('<B1-Motion>', self.pan)
        self.bind('<MouseWheel>', self.zoom)
        self.bind('<Button-4>', self.zoom)
        self.bind('<Button-5>', self.zoom)
        self.bind('<Double-Button-1>', lambda _: self.fit_to_window())

    def set_image(self, image):
        # Replacing the raster never changes the shared camera, even when the
        # newly snapped image has a different resolution or aspect ratio.
        self.original_image = image
        self.update_view()

    def fit_to_window(self):
        self.camera.fit()

    def geometry_of_image(self):
        camera = self.camera
        width, height = (v * camera.scale for v in camera.scene_size)
        return ((self.winfo_width() - width) / 2 + camera.offset_x,
                (self.winfo_height() - height) / 2 + camera.offset_y, width, height)

    def update_view(self):
        self.delete('all')
        if self.original_image is None:
            return
        x, y, width, height = self.geometry_of_image()
        left, top = max(0, math.ceil(x)), max(0, math.ceil(y))
        right = min(self.winfo_width(), math.ceil(x + width))
        bottom = min(self.winfo_height(), math.ceil(y + height))
        if right <= left or bottom <= top:
            return
        image = self.original_image
        # Each original raster is sampled directly into the same scene rect.
        # Only render the visible region, so deep zoom stays bounded in memory.
        box = (max(0, (left - x) * image.width / width),
               max(0, (top - y) * image.height / height),
               min(image.width, (right - x) * image.width / width),
               min(image.height, (bottom - y) * image.height / height))
        tile = image.resize((right - left, bottom - top), Image.Resampling.NEAREST, box=box)
        tile = ZoomableImage._composite_for_display(tile)
        self._photo = ImageTk.PhotoImage(tile)
        self.create_image(left, top, anchor='nw', image=self._photo)

    def on_resize(self, _event):
        if self.camera.auto_fit:
            self.camera.fit()
        else:
            self.camera.redraw()

    def start_pan(self, event):
        if self.pick_callback is not None:
            self._pan_start = None
            color = self.color_at(event.x, event.y)
            if color is not None:
                self.pick_callback(color)
            return
        self._pan_start = (event.x, event.y, self.camera.offset_x, self.camera.offset_y)
        self.camera.auto_fit = False

    def color_at(self, x, y):
        if self.original_image is None:
            return None
        left, top, width, height = self.geometry_of_image()
        u, v = (x - left) / width, (y - top) / height
        if not (0 <= u < 1 and 0 <= v < 1):
            return None
        image = self.original_image
        pixel = image.convert('RGBA').getpixel((int(u * image.width), int(v * image.height)))
        return pixel[:3] if pixel[3] else None

    def pan(self, event):
        if self._pan_start is None:
            return
        x, y, ox, oy = self._pan_start
        self.camera.offset_x, self.camera.offset_y = ox + event.x - x, oy + event.y - y
        self.camera.redraw()

    def zoom(self, event):
        camera = self.camera
        fine = bool(getattr(event, 'state', 0) & 1)
        factor = 1.02 if fine else 1.1
        if getattr(event, 'num', None) == 5 or getattr(event, 'delta', 0) < 0:
            factor = 1 / factor
        scale = max(.0001, min(128.0, camera.scale * factor))
        ratio = scale / camera.scale
        x, y = event.x - self.winfo_width() / 2, event.y - self.winfo_height() / 2
        camera.offset_x = x - (x - camera.offset_x) * ratio
        camera.offset_y = y - (y - camera.offset_y) * ratio
        camera.scale = scale
        camera.auto_fit = False
        camera.redraw()
