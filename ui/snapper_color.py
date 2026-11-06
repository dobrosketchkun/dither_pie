"""HSV/RGB/hex palette editor with sampling from the snapper previews."""
import colorsys
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from .palettes import HSVColorPickerDialog, PLANE_SIZE


class SnapColorDialog(HSVColorPickerDialog):
    def __init__(self, parent, initial, *, can_delete=False):
        super().__init__(parent)
        self.title('Edit palette color' if can_delete else 'Choose palette color')
        self.geometry('640x460')
        self.action = None
        controls = self.preview_box.master
        ctk.CTkButton(controls, text='Pick from image', command=lambda: parent._start_color_pick(self)).grid(
            row=3, column=0, columnspan=4, sticky='ew', padx=5, pady=(16, 5))
        if can_delete:
            ctk.CTkButton(controls, text='Delete color', command=self._delete).grid(
                row=4, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
        ctk.CTkButton(self, text='Cancel', command=self.destroy).pack(side='bottom', pady=(0, 6))
        self.bind('<Escape>', lambda _: self.destroy())
        self.set_rgb(initial)

    def set_rgb(self, rgb):
        h, self.sat, self.val = colorsys.rgb_to_hsv(*(v / 255 for v in rgb))
        self.hue = h * 360
        self.hue_slider.set(self.hue)
        self.update_color_plane()
        self.update_preview()
        self.update_color_reps()

    def get_rgb(self):
        return tuple(round(v * 255) for v in colorsys.hsv_to_rgb(self.hue / 360, self.sat, self.val))

    def update_color_plane(self):
        # Vectorized HSV plane; avoid the legacy picker's nested Python loops.
        saturation = np.linspace(0, 1, PLANE_SIZE)[None, :, None]
        value = np.linspace(1, 0, PLANE_SIZE)[:, None, None]
        hue = np.asarray(colorsys.hsv_to_rgb(self.hue / 360, 1, 1))
        pixels = np.rint(255 * value * (1 - saturation + saturation * hue)).astype(np.uint8)
        self.plane_image = Image.fromarray(pixels, 'RGB')
        self.plane_photo = ImageTk.PhotoImage(self.plane_image)
        self.plane_canvas.delete('all')
        self.circle_id = None
        self.plane_canvas.create_image(0, 0, anchor='nw', image=self.plane_photo)
        self.update_circle()

    def _delete(self):
        self.action = 'delete'
        self.destroy()
