# dithering_app.py
"""
A GUI application for image and video dithering with pixelization effects.
Uses dithering_lib for pixelization, color reduction, and various dithering algorithms.
"""

import sys
import os
import glob
import shutil
import random
import json
import subprocess
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

# Import from our library file
from dithering_lib import (
    DitherMode,
    DitherUtils,
    ImageDitherer,
    ColorReducer
)

import tempfile
import uuid

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# -------------------- GUI: ZoomableImage --------------------

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


# -------------------- GUI: PalettePreview --------------------

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


# -------------------- GUI: PaletteDialog --------------------

class PaletteDialog(ctk.CTkToplevel):
    """
    Dialog that shows different generated or custom palettes for user selection.
    """
    def __init__(self, parent, image: Image.Image, custom_palettes, save_callback):
        super().__init__(parent)
        self.title("Select Palette")
        self.geometry("600x600")
        self.image = image
        self.selected_palette = None
        self.selected_palette_name = None
        self.custom_palettes = custom_palettes
        self.save_callback = save_callback

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.palettes = self.generate_palettes()
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_palette_options()

        self.custom_buttons_frame = ctk.CTkFrame(self)
        self.custom_buttons_frame.pack(pady=10, fill='x')

        self.create_custom_palette_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Create Custom Palette",
            command=self.create_custom_palette
        )
        self.create_custom_palette_button.pack(side="left", padx=5, fill='x', expand=True)

        self.import_palette_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Import from lospec.com",
            command=self.import_from_lospec
        )
        self.import_palette_button.pack(side="left", padx=5, fill='x', expand=True)

        self.create_from_image_button = ctk.CTkButton(
            self.custom_buttons_frame,
            text="Create from Image",
            command=self.create_palette_from_image
        )
        self.create_from_image_button.pack(side="left", padx=5, fill='x', expand=True)

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(fill="x", padx=10, pady=5)

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="left", padx=5)

        self.ok_button = ctk.CTkButton(self.button_frame, text="OK", command=self.confirm)
        self.ok_button.pack(side="right", padx=5)

    def generate_palettes(self):
        pals = []
        try:
            num_colors = int(self.master.colors_entry.get())
        except:
            num_colors = 16

        # Median Cut
        mc = ColorReducer.reduce_colors(self.image, num_colors)
        pals.append(("Median Cut", mc))

        # K-means variants
        km1 = ColorReducer.generate_kmeans_palette(self.image, num_colors, random_state=42)
        km2 = ColorReducer.generate_kmeans_palette(self.image, num_colors, random_state=123)
        pals.append(("K-means (Variant 1)", km1))
        pals.append(("K-means (Variant 2)", km2))

        # Uniform
        up = ColorReducer.generate_uniform_palette(num_colors)
        pals.append(("Uniform", up))

        return pals

    def create_palette_options(self):
        self.selected_var = tk.StringVar(value="Median Cut")
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        for name, palette in self.palettes:
            fr = ctk.CTkFrame(self.scroll_frame)
            fr.pack(fill="x", padx=5, pady=5)
            radio = ctk.CTkRadioButton(fr, text=name, variable=self.selected_var, value=name)
            radio.pack(side="left", padx=10)
            prev = PalettePreview(fr, palette)
            prev.pack(side="right", padx=10, fill="x", expand=True)

        for name, palette in self.custom_palettes:
            fr = ctk.CTkFrame(self.scroll_frame)
            fr.pack(fill="x", padx=5, pady=5)
            radio = ctk.CTkRadioButton(fr, text=name, variable=self.selected_var, value=name)
            radio.pack(side="left", padx=10)
            prev = PalettePreview(fr, palette)
            prev.pack(side="right", padx=10, fill="x", expand=True)

    def create_custom_palette(self):
        CustomPaletteCreator(self, self.custom_palettes, self.save_callback, self.refresh_palettes)

    def create_palette_from_image(self):
        fp = filedialog.askopenfilename(
            parent=self,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                       ("All files", "*.*")]
        )
        if not fp:
            return
        try:
            new_img = Image.open(fp)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}", parent=self)
            return

        try:
            desired = int(self.master.colors_entry.get())
            if desired <= 0:
                raise ValueError
        except:
            desired = 16

        arr_full = np.array(new_img.convert('RGB'))
        all_pixels = arr_full.reshape(-1, 3)
        unique_pixels = np.unique(all_pixels, axis=0)
        unique_count = unique_pixels.shape[0]

        if unique_count < desired:
            n = unique_count
        else:
            n = desired
        if n < 1:
            n = 1

        if len(all_pixels) > 10000:
            idx = np.random.choice(len(all_pixels), 10000, replace=False)
            small = all_pixels[idx]
        else:
            small = all_pixels

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(small)
        centers = kmeans.cluster_centers_.astype(int)
        kpal = [tuple(v) for v in centers]

        from_img_preview = PaletteImagePreviewDialog(self, kpal, fp, used_clusters=n)
        self.wait_window(from_img_preview)
        if from_img_preview.choose_another:
            self.create_palette_from_image()
            return
        elif from_img_preview.use_result:
            bn = os.path.basename(fp)
            pname = "From imported image"
            # remove old entry if name already used
            self.palettes = [(nm, pl) for (nm, pl) in self.palettes if nm != pname]
            self.palettes.insert(0, (pname, kpal))
            self.create_palette_options()
            self.selected_var.set(pname)

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
            import urllib.request
            with urllib.request.urlopen(json_url) as resp:
                data = resp.read()
                pjson = json.loads(data)
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download or parse palette JSON:\n{e}", parent=self)
            return
        try:
            name = pjson['name']
            colors = pjson['colors']
            def hx2rgb(hx: str) -> Tuple[int,int,int]:
                hx = hx.lstrip('#')
                return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))
            rgb_cols = [hx2rgb(f"#{c}") for c in colors]
        except KeyError as e:
            messagebox.showerror("JSON Error", f"Missing key in palette JSON: {e}", parent=self)
            return
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse palette JSON:\n{e}", parent=self)
            return
        ex_names = [nm for nm,_ in self.palettes] + [nm for nm,_ in self.custom_palettes]
        if name in ex_names:
            messagebox.showerror("Duplicate Palette", f"A palette named '{name}' already exists.", parent=self)
            return
        self.custom_palettes.append((name, rgb_cols))
        self.save_callback()
        self.create_palette_options()
        self.selected_var.set(name)
        messagebox.showinfo("Success", f"Palette '{name}' imported successfully.", parent=self)

    def get_selected_palette(self):
        sname = self.selected_var.get()
        for nm, pal in self.palettes:
            if nm == sname:
                self.selected_palette_name = nm
                return pal
        for nm, pal in self.custom_palettes:
            if nm == sname:
                self.selected_palette_name = nm
                return pal
        return None

    def cancel(self):
        self.selected_palette = None
        self.selected_palette_name = None
        self.destroy()

    def confirm(self):
        self.selected_palette = self.get_selected_palette()
        if self.selected_palette:
            self.master.last_used_palette = self.selected_palette
        self.destroy()


# -------------------- GUI: PaletteImagePreviewDialog --------------------

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


# -------------------- GUI: HSVColorPickerDialog --------------------

import colorsys

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
        import colorsys
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


# -------------------- GUI: CustomPaletteCreator --------------------

class CustomPaletteCreator(ctk.CTkToplevel):
    """
    A small editor that lets you create a custom palette by picking multiple HSV colors.
    """
    def __init__(self, parent, custom_palettes, save_callback, refresh_callback):
        super().__init__(parent)
        self.title("Create Custom Palette")
        self.geometry("500x400")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.custom_palettes = custom_palettes
        self.save_callback = save_callback
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
        pick.wait_window()
        if pick.selected_color is not None:
            self.colors.append(pick.selected_color)
            self.update_palette_display()

    def delete_color(self, index: int):
        if 0 <= index < len(self.colors):
            del self.colors[index]
            self.update_palette_display()

    def save_palette(self):
        if not self.colors:
            messagebox.showwarning("No Colors", "Please add at least one color to the palette.", parent=self)
            return
        pname = simpledialog.askstring("Palette Name", "Enter a name for the custom palette:", parent=self)
        if not pname:
            return
        enames = [n for n,_ in self.custom_palettes]
        if pname in enames:
            messagebox.showerror("Duplicate Name", "A palette with this name already exists.", parent=self)
            return
        self.custom_palettes.append((pname, self.colors.copy()))
        self.save_callback()
        self.refresh_callback()
        self.destroy()


# -------------------- GUI: Main App --------------------

class App(ctk.CTk):
    """
    The main application window that includes a sidebar for options
    and the main area for image display. Also holds references to
    the currently loaded image, dithering settings, etc.
    """
    def __init__(self):
        super().__init__()
        self.title("Image Dithering Tool")
        self.geometry("1200x800")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.sidebar.grid_rowconfigure(16, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.image_viewer = ZoomableImage(self.main_frame, bg="gray20", highlightthickness=0)
        self.image_viewer.grid(row=0, column=0, sticky='nsew')

        self.create_controls()

        # Variables for images / states
        self.current_image = None
        self.original_filename = None
        self.dithered_image = None
        self.pixelized_image = None
        self.display_state = "pixelized"
        self.current_palette_name = None

        self.palette_file = "palette.json"
        self.custom_palettes = []
        self.load_custom_palettes()

        self.is_video = False
        self.video_path = None
        self.last_used_palette = None

    def create_controls(self):
        row = 0
        self.open_button = ctk.CTkButton(self.sidebar, text="Open Image/Video", command=self.open_image)
        self.open_button.grid(row=row, column=0, padx=20, pady=(10,5), sticky='ew')
        row += 1

        self.mode_label = ctk.CTkLabel(self.sidebar, text="Dither Mode:")
        self.mode_label.grid(row=row, column=0, padx=20, pady=(10,0), sticky='w')
        row += 1

        self.dither_mode = ctk.CTkOptionMenu(self.sidebar,
                                             values=[m.value for m in DitherMode])
        self.dither_mode.grid(row=row, column=0, padx=20, pady=(0,10), sticky='ew')
        self.dither_mode.set(DitherMode.BAYER4x4.value)
        row += 1

        self.colors_label = ctk.CTkLabel(self.sidebar, text="Number of Colors:")
        self.colors_label.grid(row=row, column=0, padx=20, pady=(10,0), sticky='w')
        row += 1

        self.colors_entry = ctk.CTkEntry(self.sidebar, placeholder_text="16")
        self.colors_entry.insert(0, "16")
        self.colors_entry.grid(row=row, column=0, padx=20, pady=(0,10), sticky='ew')
        row += 1

        self.auto_pixelize_var = tk.BooleanVar(value=True)
        self.auto_pixelize_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Automatic Pixelization",
            variable=self.auto_pixelize_var,
            command=self.toggle_auto_pixelization
        )
        self.auto_pixelize_check.grid(row=row, column=0, padx=20, pady=(0,10), sticky='w')
        row += 1

        self.max_size_label = ctk.CTkLabel(self.sidebar, text="Maximum Size:")
        self.max_size_entry = ctk.CTkEntry(self.sidebar, placeholder_text="640")
        self.max_size_entry.insert(0, "640")

        self.max_size_label.grid(row=row, column=0, padx=20, pady=(0,0), sticky='w')
        self.max_size_entry.grid(row=row+1, column=0, padx=20, pady=(0,10), sticky='ew')
        self.max_size_label.grid_remove()
        self.max_size_entry.grid_remove()
        row += 2

        self.apply_button = ctk.CTkButton(self.sidebar, text="Apply Dithering", command=self.show_palette_dialog)
        self.apply_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        # Gamma Correction
        self.use_gamma_var = tk.BooleanVar(value=False)
        self.gamma_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Use Gamma Correction",
            variable=self.use_gamma_var
        )
        self.gamma_check.grid(row=row, column=0, padx=20, pady=(0,10), sticky='w')
        row += 1

        self.save_button = ctk.CTkButton(self.sidebar, text="Save Image", command=self.save_image)
        self.save_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        self.pixelize_button = ctk.CTkButton(self.sidebar, text="Pixelize", command=self.pixelize_image)
        self.pixelize_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        self.pixelize_button = ctk.CTkButton(self.sidebar, text="Neural pixelize", command=self.pixelize_image_ai)
        self.pixelize_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        self.reset_button = ctk.CTkButton(self.sidebar, text="Fit to Window", command=self.fit_to_window)
        self.reset_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        self.toggle_button = ctk.CTkButton(self.sidebar, text="Toggle View", command=self.toggle_view)
        self.toggle_button.grid(row=row, column=0, padx=20, pady=10, sticky='ew')
        row += 1

        self.random_frame_button = ctk.CTkButton(
            self.sidebar, text="New Random Frame", command=self.load_random_frame
        )
        self.random_frame_button.grid(row=row, column=0, padx=20, pady=(10,5), sticky='ew')
        row += 1
        self.random_frame_button.grid_remove()

        self.apply_video_button = ctk.CTkButton(
            self.sidebar, text="Apply to Video", command=self.apply_to_video
        )
        self.apply_video_button.grid(row=row, column=0, padx=20, pady=(10,5), sticky='ew')
        row += 1
        self.apply_video_button.grid_remove()

        for i in range(row, 17):
            self.sidebar.grid_rowconfigure(i, weight=1)

    def toggle_auto_pixelization(self):
        if self.auto_pixelize_var.get():
            self.max_size_label.grid_remove()
            self.max_size_entry.grid_remove()
        else:
            self.max_size_label.grid()
            self.max_size_entry.grid()

    def open_image(self):
        fp = filedialog.askopenfilename(
            filetypes=[
                ("Image/Video files", "*.png *.jpg *.jpeg *.gif *.bmp *.mp4 *.mkv *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        if not fp:
            return
        ext = os.path.splitext(fp)[1].lower()
        vexts = [".mp4", ".mkv", ".avi", ".mov"]
        if ext in vexts:
            self.is_video = True
            self.video_path = fp
            self.random_frame_button.grid()
            self.apply_video_button.grid()
            self.load_random_frame()
        else:
            self.is_video = False
            self.video_path = None
            self.random_frame_button.grid_remove()
            self.apply_video_button.grid_remove()
            try:
                self.current_image = Image.open(fp)
                self.original_filename = fp
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
                self.last_used_palette = None

    def load_random_frame(self):
        if not self.is_video or not self.video_path:
            return
        try:
            cmd = [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", self.video_path
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            total_frames_str = proc.stdout.strip()
            total_frames = int(total_frames_str) if total_frames_str else 0
            if total_frames < 1:
                raise ValueError("Failed to get frame count.")

            idx = random.randint(1, total_frames-1)
            tmp_frame = "tmp_preview_frame.png"
            ext_cmd = [
                "ffmpeg", "-y", "-i", self.video_path,
                "-vf", f"select='eq(n,{idx})'",
                "-vframes", "1",
                tmp_frame
            ]
            subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            self.current_image = Image.open(tmp_frame)
            self.original_filename = self.video_path
            if self.auto_pixelize_var.get():
                self.pixelize_image(auto=True)
            else:
                self.pixelized_image = self.current_image.convert('RGB')
                self.display_state = "pixelized"
                self.image_viewer.set_image(self.pixelized_image)
                self.fit_to_window()
                self.dithered_image = None
                self.last_used_palette = None

            os.remove(tmp_frame)

        except Exception as e:
            messagebox.showerror("Video Error", f"Failed to load random frame:\n{e}")

    def show_palette_dialog(self):
        if not self.pixelized_image:
            messagebox.showwarning("No Pixelized Image", "Please pixelize the image first.")
            return
        dlg = PaletteDialog(self, self.pixelized_image, self.custom_palettes, self.save_custom_palettes)
        self.wait_window(dlg)
        if dlg.selected_palette:
            try:
                nc = int(self.colors_entry.get())
                if nc <= 0:
                    raise ValueError
            except:
                messagebox.showerror("Invalid Number of Colors", "Please enter a valid positive integer.")
                return
            try:
                dm = DitherMode(self.dither_mode.get())
            except:
                messagebox.showerror("Invalid Dither Mode", "Please select a valid dither mode.")
                return

            self.current_palette_name = dlg.selected_palette_name or "UnknownPalette"
            use_gamma = self.use_gamma_var.get()
            ditherer = ImageDitherer(num_colors=nc, dither_mode=dm,
                                     palette=dlg.selected_palette, use_gamma=use_gamma)
            try:
                self.dithered_image = ditherer.apply_dithering(self.pixelized_image)
                self.last_used_palette = dlg.selected_palette
            except Exception as e:
                messagebox.showerror("Dithering Error", f"An error occurred:\n{e}")
                return
            self.display_state = "dithered"
            self.image_viewer.set_image(self.dithered_image)
            self.fit_to_window()

    def pixelize_image(self, auto=False):
        if not self.current_image:
            if not auto:
                messagebox.showwarning("No Image", "Please open an image (or video) first.")
            return

        if self.auto_pixelize_var.get() or auto:
            mx = 640
        else:
            try:
                mx = int(self.max_size_entry.get())
                if mx <= 0:
                    raise ValueError
            except:
                messagebox.showerror("Invalid Maximum Size", "Please enter a valid positive integer.")
                return

        ratio = self.current_image.width / self.current_image.height
        if self.current_image.width >= self.current_image.height:
            nh = mx
            nw = int(mx * ratio)
        else:
            nw = mx
            nh = int(mx / ratio)

        nw = (nw // 2) * 2
        nh = (nh // 2) * 2

        resized = self.current_image.resize((nw, nh), Image.Resampling.NEAREST)
        final = resized.convert('RGB')
        self.pixelized_image = final
        self.display_state = "pixelized"
        self.image_viewer.set_image(self.pixelized_image)
        self.fit_to_window()
        self.dithered_image = None
        self.last_used_palette = None

        # ---- NEW: record that we used the "regular" pixelization method ----
        self.last_pixelization_method = "regular"

        if not auto:
            messagebox.showinfo("Pixelization Complete", "Image has been pixelized.")


    def pixelize_image_ai(self, auto=False):
        import torch
        from models.pixelization import Model, resize_image

        if not self.current_image:
            if not auto:
                messagebox.showwarning("No Image", "Please open an image (or video) first.")
            return

        if self.auto_pixelize_var.get() or auto:
            mx = 640
        else:
            try:
                mx = int(self.max_size_entry.get())
                if mx <= 0:
                    raise ValueError
            except:
                messagebox.showerror("Invalid Maximum Size", "Please enter a valid positive integer.")
                return

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_id = str(uuid.uuid4())
            temp_input = os.path.join(tmpdir, f"temp_input_{temp_id}.png")
            temp_output = os.path.join(tmpdir, f"temp_output_{temp_id}.png")
            temp_resized_input = os.path.join(tmpdir, f"temp_resized_input_{temp_id}.png")

            self.current_image.save(temp_input)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            m = Model(device=device_type)
            m.load()

            resize_image(temp_input, temp_resized_input, mx * 4)
            m.pixelize(temp_resized_input, temp_output)
            resize_image(temp_output, temp_output, mx)

            final_img = Image.open(temp_output).convert('RGB')

        self.pixelized_image = final_img
        self.display_state = "pixelized"
        self.image_viewer.set_image(self.pixelized_image)
        self.fit_to_window()
        self.dithered_image = None
        self.last_used_palette = None

        # ---- NEW: record that we used the "neural" pixelization method ----
        self.last_pixelization_method = "neural"

        if not auto:
            messagebox.showinfo("Pixelization Complete", "Image has been AI-pixelized.")

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
            safe_pn = self.current_palette_name.replace(' ', '_')
            parts.append(safe_pn)

        try:
            nc = int(self.colors_entry.get())
        except:
            nc = 16
        parts.append(f"{nc}colors")

        if self.use_gamma_var.get():
            parts.append("gamma")

        default_filename = '_'.join(parts) + ".png"

        fp = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if fp:
            try:
                image_to_save.save(fp)
                messagebox.showinfo("Image Saved", f"Image saved as: {fp}")
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
            for p in data:
                nm = p['name']
                def hx2rgb(hx:str) -> tuple:
                    hx = hx.lstrip('#')
                    return tuple(int(hx[i:i+2],16) for i in (0,2,4))
                cols = [hx2rgb(c) for c in p['colors']]
                self.custom_palettes.append((nm, cols))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load custom palettes:\n{e}")
            self.custom_palettes = []

    def save_custom_palettes(self):
        data = []
        for nm, cols in self.custom_palettes:
            def rgb2hx(rgb:tuple) -> str:
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            hxcols = [rgb2hx(c) for c in cols]
            data.append({"name": nm, "colors": hxcols})
        try:
            with open(self.palette_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save custom palettes:\n{e}")

    def apply_to_video(self):
        if not self.is_video or not self.video_path:
            messagebox.showwarning("Not Video", "No video to process.")
            return
        if self.dithered_image is None and self.pixelized_image is None:
            messagebox.showwarning("No Dither", "Please dither or pixelize at least once.")
            return

        if self.original_filename:
            base_name = os.path.splitext(os.path.basename(self.original_filename))[0]
        else:
            base_name = "untitled"

        parts = [base_name]
        if self.dithered_image is not None:
            parts.append("dithered")
            parts.append(self.dither_mode.get())
        else:
            parts.append("pixelized")

        if self.current_palette_name:
            sp = self.current_palette_name.replace(' ', '_')
            parts.append(sp)

        try:
            nc = int(self.colors_entry.get())
        except:
            nc = 16
        parts.append(f"{nc}colors")

        if self.use_gamma_var.get():
            parts.append("gamma")

        default_out = '_'.join(parts) + ".mp4"

        out_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=default_out,
            filetypes=[("MP4 video files","*.mp4"),("All files","*.*")]
        )
        if not out_path:
            return

        try:
            tmp_dir = "frames_tmp"
            os.makedirs(tmp_dir, exist_ok=True)

            # Extract frames
            extract_cmd = [
                "ffmpeg", "-y", "-i", self.video_path,
                os.path.join(tmp_dir, "frame_%05d.png")
            ]
            subprocess.run(extract_cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)

            # Get FPS
            fps_cmd = [
                "ffprobe","-v","error","-select_streams","v:0",
                "-show_entries","stream=r_frame_rate",
                "-of","default=nokey=1:noprint_wrappers=1",
                self.video_path
            ]
            fps_proc = subprocess.run(fps_cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, check=True)
            raw = fps_proc.stdout.strip()
            if raw and "/" in raw:
                num, den = raw.split("/")
                try:
                    float_fps = float(num)/float(den)
                except:
                    float_fps = 30.0
            else:
                float_fps = 30.0

            # Build palette if needed
            if self.last_used_palette is not None:
                palette_for_video = self.last_used_palette
            else:
                # fallback: reduce from current pixelized image
                palette_for_video = ColorReducer.reduce_colors(self.pixelized_image, nc)

            try:
                dm = DitherMode(self.dither_mode.get())
            except:
                dm = DitherMode.BAYER4x4

            use_gamma = self.use_gamma_var.get()
            dith = ImageDitherer(num_colors=nc, dither_mode=dm,
                                 palette=palette_for_video, use_gamma=use_gamma)

            # Decide if the user last used neural pixelization or not.
            # "regular" means nearest-neighbor, "neural" means AI approach
            last_pixel_method = getattr(self, "last_pixelization_method", "regular")

            frame_files = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png")))
            total_frames = len(frame_files)

            for i, fpn in enumerate(frame_files, start=1):
                try:
                    frm = Image.open(fpn)

                    if not self.auto_pixelize_var.get():
                        try:
                            mxv = int(self.max_size_entry.get())
                            if mxv <= 0: raise ValueError
                        except:
                            mxv = 640
                    else:
                        mxv = 640

                    # Pixelize using whichever method was last used by the user:
                    if last_pixel_method == "neural":
                        # same steps as pixelize_image_ai, but in a simplified way
                        # For brevity, let's do a function call:
                        frm = self._pixelize_frame_neural(frm, mxv)
                    else:
                        frm = self._pixelize_frame_regular(frm, mxv)

                    # Then dither if the user had clicked "Apply dithering"
                    dimg = dith.apply_dithering(frm)
                    dimg.save(fpn)

                except Exception as e:
                    print(f"\nERROR on frame {i} / {fpn}: {e}", file=sys.stderr)
                    continue

                prog = float(i) / total_frames
                bar_len = 30
                filled = int(bar_len * prog)
                bar = '#' * filled + '-' * (bar_len - filled)
                sys.stdout.write(f"\rDithering frames: [{bar}] {i}/{total_frames}")
                sys.stdout.flush()

            print()

            # Re-encode
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{float_fps:.5f}",
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-i", self.video_path,
                "-map", "0:v",
                "-map", "1:a?",
                "-map", "1:s?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-c:s", "copy",
                "-r", f"{float_fps:.5f}",
                out_path
            ]
            subprocess.run(encode_cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)

            shutil.rmtree(tmp_dir)
            messagebox.showinfo("Video Complete", f"Video successfully dithered and saved to: {out_path}")

        except Exception as e:
            print(f"\nFailed to process video:\n{e}")
            messagebox.showerror("Video Processing Error", f"Failed to process video:\n{e}")
            

    def compute_even_dimensions(self, orig_w: int, orig_h: int, max_size: int) -> Tuple[int, int]:
        """
        Compute target dimensions such that the smaller side is close to max_size and both dimensions are even.
        This ensures that when resizing, no cropping occurs and the final dimensions are acceptable for libx264.
        """
        if orig_w >= orig_h:
            # Landscape: use max_size as the target height (adjusted to be even)
            target_h = max_size if max_size % 2 == 0 else max_size - 1
            target_w = int(round((orig_w / orig_h) * target_h))
            if target_w % 2 != 0:
                target_w += 1  # Adjust upward to make it even.
        else:
            # Portrait: use max_size as the target width (adjusted to be even)
            target_w = max_size if max_size % 2 == 0 else max_size - 1
            target_h = int(round((orig_h / orig_w) * target_w))
            if target_h % 2 != 0:
                target_h += 1  # Adjust upward to make it even.
        return target_w, target_h

    def _pixelize_frame_regular(self, frame_pil: Image.Image, max_size: int) -> Image.Image:
        """
        Regular pixelization using nearest-neighbor.
        Computes even target dimensions before resizing.
        """
        orig_w, orig_h = frame_pil.size
        target_w, target_h = self.compute_even_dimensions(orig_w, orig_h, max_size)
        resized = frame_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
        return resized.convert('RGB')

    def _pixelize_frame_neural(self, frame_pil: Image.Image, max_size: int) -> Image.Image:
        """
        Neural pixelization using the AI model.
        After the neural process, the final image is resized to even dimensions.
        """
        import torch
        from models.pixelization import Model, resize_image

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_id = str(uuid.uuid4())
            temp_input = os.path.join(tmpdir, f"temp_input_{temp_id}.png")
            temp_output = os.path.join(tmpdir, f"temp_output_{temp_id}.png")
            temp_resized_input = os.path.join(tmpdir, f"temp_resized_input_{temp_id}.png")

            # Save the original frame temporarily.
            frame_pil.save(temp_input)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            m = Model(device=device_type)
            m.load()

            # Resize for neural processing.
            resize_image(temp_input, temp_resized_input, max_size * 4)
            m.pixelize(temp_resized_input, temp_output)
            
            # Load the result and then compute even dimensions.
            final_img = Image.open(temp_output).convert('RGB')
            orig_w, orig_h = final_img.size
            target_w, target_h = self.compute_even_dimensions(orig_w, orig_h, max_size)
            final_img = final_img.resize((target_w, target_h), Image.Resampling.NEAREST)
        return final_img


# -------------------- Main Entry Point --------------------

def main():
    """Launch the GUI application."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
