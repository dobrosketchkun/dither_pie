"""Palettes components."""

import os
import colorsys
import tkinter as tk
from tkinter import messagebox, simpledialog, colorchooser
import customtkinter as ctk
from PIL import Image, ImageTk
from typing import List, Tuple, Callable

__all__ = ['PalettePreview', 'ColorPickerGrid', 'HSVColorPickerDialog', 'CustomPaletteCreator', 'PaletteImagePreviewDialog']


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
