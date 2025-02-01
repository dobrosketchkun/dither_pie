import json
import sys
import os
import math
import glob
import shutil
import random
import subprocess
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from PIL import Image, ImageTk
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog

# -------------------- Enumerations --------------------

class DitherMode(Enum):
    NONE = "none"
    BAYER2x2 = "bayer2x2"
    BAYER4x4 = "bayer4x4"
    BAYER8x8 = "bayer8x8"
    BAYER16x16 = "bayer16x16"

# -------------------- Dither Utility --------------------

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
            raise ValueError("Unsupported dither mode.")

    # -------------------- NEW: Gamma Correction Methods --------------------
    @staticmethod
    def srgb_to_linear(c: np.ndarray) -> np.ndarray:
        """
        c: float32 array in [0..1], shape (...,3) or just (...).
        Returns float32 array in linear space [0..1].
        """
        # piecewise transformation
        # np.where() handles vectorized operations
        low = (c <= 0.04045)
        out = np.empty_like(c, dtype=np.float32)
        out[low] = (c[low] / 12.92)
        out[~low] = (((c[~low] + 0.055) / 1.055)**2.4)
        return out

    @staticmethod
    def linear_to_srgb(c: np.ndarray) -> np.ndarray:
        """
        c: float32 array in [0..1], shape (...,3) or just (...).
        Returns float32 array in sRGB space [0..1].
        """
        low = (c <= 0.0031308)
        out = np.empty_like(c, dtype=np.float32)
        out[low] = (c[low] * 12.92)
        out[~low] = (1.055*(c[~low]**(1.0/2.4)) - 0.055)
        return out

# -------------------- ColorReducer --------------------

class ColorReducer:
    @staticmethod
    def find_dominant_channel(colors: List[Tuple[int, int, int]]) -> int:
        rng = [0,0,0]
        for channel in range(3):
            vals = [c[channel] for c in colors]
            mn, mx = min(vals), max(vals)
            rng[channel] = mx - mn
        return rng.index(max(rng))

    @staticmethod
    def median_cut(colors: List[Tuple[int,int,int]], depth:int) -> List[Tuple[int,int,int]]:
        if depth == 0 or len(colors)==0:
            if not colors:
                return [(0,0,0)]
            avg = tuple(int(sum(c)/len(c)) for c in zip(*colors))
            return [avg]

        channel = ColorReducer.find_dominant_channel(colors)
        colors.sort(key=lambda x: x[channel])
        mid = len(colors)//2
        return ColorReducer.median_cut(colors[:mid], depth-1)+ColorReducer.median_cut(colors[mid:], depth-1)

    @staticmethod
    def reduce_colors(image: Image.Image, num_colors: int) -> List[Tuple[int,int,int]]:
        image = image.convert('RGB')
        unique_cols = list(set(image.getdata()))
        if num_colors < 1: num_colors=1
        depth = int(math.log2(num_colors)) if num_colors>1 else 0
        return ColorReducer.median_cut(unique_cols, depth)

# -------------------- Image Ditherer --------------------

class ImageDitherer:
    def __init__(self, num_colors=16, dither_mode:DitherMode=DitherMode.BAYER4x4, palette:List[Tuple[int,int,int]]=None, use_gamma:bool=False):
        self.num_colors = num_colors
        self.dither_mode = dither_mode
        self.palette = palette
        self.use_gamma = use_gamma
        self.threshold_matrix = DitherUtils.get_threshold_matrix(dither_mode)

    def apply_dithering(self, image:Image.Image)->Image.Image:
        """
        If use_gamma=True, this method will:
          - Convert sRGB -> linear (0..1)
          - Scale up to 0..255 in linear 8-bit for palette building / indexing
          - Dither in that linear domain
          - Convert the final back from linear -> sRGB
        """
        # Convert image to array in 0..255
        arr_srgb_8 = np.array(image.convert('RGB'), dtype=np.uint8)

        if self.use_gamma:
            # Step 1: sRGB 8-bit -> float [0..1], then to linear
            arr_01 = arr_srgb_8.astype(np.float32)/255.0
            arr_lin_01 = DitherUtils.srgb_to_linear(arr_01)
            # Scale back to 0..255 but linear-luminance-coded
            arr_lin_8 = np.clip((arr_lin_01*255.0),0,255).astype(np.uint8)

            # If no palette, generate in linear domain
            if self.palette is None:
                # Create a temp Image from linear-coded 8-bit to pass to color reducer
                temp_img_lin = Image.fromarray(arr_lin_8, 'RGB')
                self.palette = ColorReducer.reduce_colors(temp_img_lin, self.num_colors)

            # Now do dithering in that linear-coded 8-bit space
            arr_for_dith = arr_lin_8
        else:
            # No gamma correction path
            if self.palette is None:
                self.palette = ColorReducer.reduce_colors(image, self.num_colors)
            arr_for_dith = arr_srgb_8

        # Build palette KDTree
        palette_arr = np.array(self.palette, dtype=np.float32)  # 8-bit sRGB or "linear-coded" 8-bit
        tree = KDTree(palette_arr)

        # Flatten
        h,w,_ = arr_for_dith.shape
        flat_pixels = arr_for_dith.reshape((-1,3)).astype(np.float32)

        # Dither logic
        if self.dither_mode == DitherMode.NONE:
            _, idx = tree.query(flat_pixels, k=1, workers=-1)
            dith_pixels = palette_arr[idx,:]
            out_arr_lin_8 = dith_pixels.reshape((h,w,3)).astype(np.uint8)
        else:
            distances, indices = tree.query(flat_pixels, k=2, workers=-1)
            dist_sq = distances**2
            dist_nearest = dist_sq[:,0]
            dist_second = dist_sq[:,1]
            total_dist = dist_nearest+dist_second
            factor = np.where(total_dist==0, 0.0, dist_nearest/total_dist)
            th_mat = self.threshold_matrix
            th_h, th_w = th_mat.shape
            t = np.tile(th_mat, ((h+th_h-1)//th_h, (w+th_w-1)//th_w))
            t = t[:h,:w]
            flat_thresh = t.flatten()
            idx_nearest = indices[:,0]
            idx_second = indices[:,1]
            use_nearest = (factor<=flat_thresh)
            final_indices = np.where(use_nearest, idx_nearest, idx_second).astype(np.int32)
            dith_pixels = palette_arr[final_indices,:]
            out_arr_lin_8 = dith_pixels.reshape((h,w,3)).astype(np.uint8)

        if self.use_gamma:
            # Convert final from linear-coded 8-bit back to sRGB
            # out_arr_lin_8 in [0..255], but linear-coded
            out_lin_01 = out_arr_lin_8.astype(np.float32)/255.0
            out_srgb_01 = DitherUtils.linear_to_srgb(np.clip(out_lin_01,0,1))
            out_srgb_8 = np.clip(out_srgb_01*255.0,0,255).astype(np.uint8)
            return Image.fromarray(out_srgb_8, 'RGB')
        else:
            return Image.fromarray(out_arr_lin_8, 'RGB')

# -------------------- GUI: ZoomableImage --------------------

class ZoomableImage(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master,**kwargs)
        self.master=master
        self.original_image:Optional[Image.Image]=None
        self.displayed_image:Optional[ImageTk.PhotoImage]=None
        self.image_id:Optional[int]=None
        self.zoom_factor=1.0
        self.offset_x=0
        self.offset_y=0
        self.pan_start_x=0
        self.pan_start_y=0

        self.bind("<ButtonPress-1>",self.start_pan)
        self.bind("<B1-Motion>",self.pan)
        self.bind("<MouseWheel>",self.zoom)
        self.bind("<Button-4>",self.zoom)
        self.bind("<Button-5>",self.zoom)
        self.bind("<Configure>", self.on_resize)

    def set_image(self, image:Image.Image):
        self.original_image=image
        self.zoom_factor=1.0
        self.offset_x=0
        self.offset_y=0
        self.update_view()

    def fit_to_window(self):
        if not self.original_image: return
        self.update_idletasks()
        cw,ch=self.winfo_width(),self.winfo_height()
        iw,ih=self.original_image.size
        wr,hr=cw/iw,ch/ih
        self.zoom_factor=min(wr,hr)
        self.offset_x=0
        self.offset_y=0
        self.update_view()

    def update_view(self):
        if not self.original_image: return
        nw=int(self.original_image.width*self.zoom_factor)
        nh=int(self.original_image.height*self.zoom_factor)
        if nw<=0 or nh<=0: return

        resized=self.original_image.resize((nw,nh), Image.Resampling.NEAREST)
        self.displayed_image=ImageTk.PhotoImage(resized)

        cw,ch=self.winfo_width(),self.winfo_height()
        x=(cw-nw)//2+self.offset_x
        y=(ch-nh)//2+self.offset_y

        self.delete("all")
        self.image_id=self.create_image(x,y,anchor='nw',image=self.displayed_image)

    def start_pan(self,event):
        self.pan_start_x=event.x-self.offset_x
        self.pan_start_y=event.y-self.offset_y

    def pan(self,event):
        self.offset_x=event.x-self.pan_start_x
        self.offset_y=event.y-self.pan_start_y
        self.update_view()

    def zoom(self,event):
        if not self.original_image: return
        if event.num==5 or event.delta<0:
            self.zoom_factor*=0.9
        else:
            self.zoom_factor*=1.1
        self.zoom_factor=max(0.01,min(30.0,self.zoom_factor))
        self.update_view()

    def on_resize(self,event):
        self.fit_to_window()

# -------------------- GUI: PalettePreview --------------------

class PalettePreview(ctk.CTkFrame):
    def __init__(self, master, palette, width=200, height=30, **kwargs):
        super().__init__(master,width=width,height=height,**kwargs)
        self.palette=palette
        self.canvas=tk.Canvas(self,width=width,height=height,highlightthickness=0)
        self.canvas.pack(fill="both",expand=True)
        self.after(100,self.draw_palette)
        self.bind("<Configure>",lambda ev: self.after(100,self.draw_palette))

    def draw_palette(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w=self.canvas.winfo_width()
        h=self.canvas.winfo_height()
        n=len(self.palette)
        if n==0: return
        seg_w=w/n
        for i,color in enumerate(self.palette):
            x1=i*seg_w
            x2=(i+1)*seg_w
            hx=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            self.canvas.create_rectangle(x1,0,x2,h,fill=hx,outline='')

# -------------------- GUI: PaletteDialog --------------------

class PaletteDialog(ctk.CTkToplevel):
    def __init__(self, parent, image:Image.Image, custom_palettes, save_callback):
        super().__init__(parent)
        self.title("Select Palette")
        self.geometry("600x600")
        self.image=image
        self.selected_palette:Optional[List[Tuple[int,int,int]]]=None
        self.selected_palette_name:Optional[str]=None
        self.custom_palettes=custom_palettes
        self.save_callback=save_callback

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.palettes=self.generate_palettes()
        self.scroll_frame=ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both",expand=True,padx=10,pady=10)

        self.create_palette_options()

        self.custom_buttons_frame=ctk.CTkFrame(self)
        self.custom_buttons_frame.pack(pady=10,fill='x')

        self.create_custom_palette_button=ctk.CTkButton(self.custom_buttons_frame,text="Create Custom Palette",
                                                        command=self.create_custom_palette)
        self.create_custom_palette_button.pack(side="left",padx=5,fill='x',expand=True)

        self.import_palette_button=ctk.CTkButton(self.custom_buttons_frame,text="Import from lospec.com",
                                                 command=self.import_from_lospec)
        self.import_palette_button.pack(side="left",padx=5,fill='x',expand=True)

        self.create_from_image_button=ctk.CTkButton(self.custom_buttons_frame,text="Create from Image",
                                                    command=self.create_palette_from_image)
        self.create_from_image_button.pack(side="left",padx=5,fill='x',expand=True)

        self.button_frame=ctk.CTkFrame(self)
        self.button_frame.pack(fill="x",padx=10,pady=5)

        self.cancel_button=ctk.CTkButton(self.button_frame,text="Cancel",command=self.cancel)
        self.cancel_button.pack(side="left",padx=5)

        self.ok_button=ctk.CTkButton(self.button_frame,text="OK",command=self.confirm)
        self.ok_button.pack(side="right",padx=5)

    def generate_palettes(self)->List[Tuple[str,List[Tuple[int,int,int]]]]:
        pals=[]
        try:
            num_colors=int(self.master.colors_entry.get())
        except:
            num_colors=16

        mc=ColorReducer.reduce_colors(self.image,num_colors)
        pals.append(("Median Cut", mc))

        km1=self.generate_kmeans_palette(self.image,num_colors,random_state=42)
        km2=self.generate_kmeans_palette(self.image,num_colors,random_state=123)
        pals.append(("K-means (Variant 1)",km1))
        pals.append(("K-means (Variant 2)",km2))

        up=self.generate_uniform_palette(num_colors)
        pals.append(("Uniform",up))
        return pals

    def generate_kmeans_palette(self, img:Image.Image, num_colors:int, random_state=42)->List[Tuple[int,int,int]]:
        arr=np.array(img.convert('RGB'))
        pix=arr.reshape(-1,3)
        if len(pix)>10000:
            idx=np.random.choice(len(pix),10000,replace=False)
            pix=pix[idx]
        kmeans=KMeans(n_clusters=num_colors,random_state=random_state)
        kmeans.fit(pix)
        centers=kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in centers]

    def generate_uniform_palette(self,num_colors)->List[Tuple[int,int,int]]:
        colors=[]
        cube=int(math.ceil(num_colors**(1/3)))
        for r in range(cube):
            for g in range(cube):
                for b in range(cube):
                    if len(colors)>=num_colors:
                        break
                    rr=int(r*255/(cube-1)) if cube>1 else 128
                    gg=int(g*255/(cube-1)) if cube>1 else 128
                    bb=int(b*255/(cube-1)) if cube>1 else 128
                    colors.append((rr,gg,bb))
        return colors[:num_colors]

    def create_palette_options(self):
        self.selected_var=tk.StringVar(value="Median Cut")
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        for name,palette in self.palettes:
            fr=ctk.CTkFrame(self.scroll_frame)
            fr.pack(fill="x",padx=5,pady=5)
            radio=ctk.CTkRadioButton(fr,text=name,variable=self.selected_var,value=name)
            radio.pack(side="left",padx=10)
            prev=PalettePreview(fr,palette)
            prev.pack(side="right",padx=10,fill="x",expand=True)

        for name,palette in self.custom_palettes:
            fr=ctk.CTkFrame(self.scroll_frame)
            fr.pack(fill="x",padx=5,pady=5)
            radio=ctk.CTkRadioButton(fr,text=name,variable=self.selected_var,value=name)
            radio.pack(side="left",padx=10)
            prev=PalettePreview(fr,palette)
            prev.pack(side="right",padx=10,fill="x",expand=True)

    def create_custom_palette(self):
        CustomPaletteCreator(self,self.custom_palettes,self.save_callback,self.refresh_palettes)

    def create_palette_from_image(self):
        fp=filedialog.askopenfilename(
            parent=self,
            filetypes=[("Image files","*.png *.jpg *.jpeg *.gif *.bmp"),("All files","*.*")]
        )
        if not fp: return
        try:
            new_img=Image.open(fp)
        except Exception as e:
            messagebox.showerror("Error",f"Failed to open image:\n{e}",parent=self)
            return

        try:
            desired=int(self.master.colors_entry.get())
            if desired<=0:
                raise ValueError
        except:
            desired=16

        arr_full=np.array(new_img.convert('RGB'))
        all_pixels=arr_full.reshape(-1,3)
        unique_pixels=np.unique(all_pixels,axis=0)
        unique_count=unique_pixels.shape[0]

        if unique_count<desired:
            n=unique_count
        else:
            n=desired
        if n<1: n=1

        if len(all_pixels)>10000:
            idx=np.random.choice(len(all_pixels),10000,replace=False)
            small=all_pixels[idx]
        else:
            small=all_pixels

        kmeans=KMeans(n_clusters=n,random_state=42)
        kmeans.fit(small)
        centers=kmeans.cluster_centers_.astype(int)
        kpal=[tuple(v) for v in centers]

        from_img_preview=PaletteImagePreviewDialog(self,kpal,fp,used_clusters=n)
        self.wait_window(from_img_preview)
        if from_img_preview.choose_another:
            self.create_palette_from_image()
            return
        elif from_img_preview.use_result:
            bn=os.path.basename(fp)
            pname="From imported image"
            self.palettes=[(nm,pl) for (nm,pl) in self.palettes if nm!=pname]
            self.palettes.insert(0,(pname,kpal))
            self.create_palette_options()
            self.selected_var.set(pname)

    def refresh_palettes(self):
        self.palettes=self.generate_palettes()
        self.create_palette_options()

    def import_from_lospec(self):
        url=simpledialog.askstring("Import Palette","Paste lospec.com Palette URL:",parent=self)
        if not url: return
        try:
            parts=url.rstrip('/').split('/')
            if len(parts)<2:
                raise ValueError("URL does not contain enough parts to extract palette name.")
            palette_slug=parts[-1]
            json_url=f'https://lospec.com/palette-list/{palette_slug}.json'
        except Exception as e:
            messagebox.showerror("Invalid URL",f"Failed to parse palette name:\n{e}",parent=self)
            return
        try:
            import urllib.request
            with urllib.request.urlopen(json_url) as resp:
                data=resp.read()
                pjson=json.loads(data)
        except Exception as e:
            messagebox.showerror("Download Error",f"Failed to download or parse palette JSON:\n{e}",parent=self)
            return
        try:
            name=pjson['name']
            colors=pjson['colors']
            def hx2rgb(hx:str)->Tuple[int,int,int]:
                hx=hx.lstrip('#')
                return tuple(int(hx[i:i+2],16) for i in (0,2,4))
            rgb_cols=[hx2rgb(f"#{c}") for c in colors]
        except KeyError as e:
            messagebox.showerror("JSON Error", f"Missing key in palette JSON: {e}",parent=self)
            return
        except Exception as e:
            messagebox.showerror("Parse Error",f"Failed to parse palette JSON:\n{e}",parent=self)
            return
        ex_names=[nm for nm,_ in self.palettes]+[nm for nm,_ in self.custom_palettes]
        if name in ex_names:
            messagebox.showerror("Duplicate Palette", f"A palette named '{name}' already exists.",parent=self)
            return
        self.custom_palettes.append((name,rgb_cols))
        self.save_callback()
        self.create_palette_options()
        self.selected_var.set(name)
        messagebox.showinfo("Success", f"Palette '{name}' imported successfully.",parent=self)

    def get_selected_palette(self)->Optional[List[Tuple[int,int,int]]]:
        sname=self.selected_var.get()
        for nm,pal in self.palettes:
            if nm==sname:
                self.selected_palette_name=nm
                return pal
        for nm,pal in self.custom_palettes:
            if nm==sname:
                self.selected_palette_name=nm
                return pal
        return None

    def cancel(self):
        self.selected_palette=None
        self.selected_palette_name=None
        self.destroy()

    def confirm(self):
        self.selected_palette=self.get_selected_palette()
        if self.selected_palette:
            self.master.last_used_palette=self.selected_palette
        self.destroy()

# -------------------- GUI: PaletteImagePreviewDialog --------------------

class PaletteImagePreviewDialog(ctk.CTkToplevel):
    def __init__(self, parent, palette:List[Tuple[int,int,int]], file_path:str, used_clusters:int):
        super().__init__(parent)
        self.title("New Palette Preview")
        self.geometry("400x180")
        self.resizable(False,False)

        self.use_result=False
        self.choose_another=False

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        bn=os.path.basename(file_path)
        info=ctk.CTkLabel(self,text=f"Generated a {used_clusters}-color palette from:\n{bn}\n\nUse this palette or pick another image?")
        info.pack(pady=(10,0))

        self.preview=PalettePreview(self,palette,width=300,height=30)
        self.preview.pack(pady=10)

        bf=ctk.CTkFrame(self)
        bf.pack(pady=5,fill='x')

        ub=ctk.CTkButton(bf,text="Use This Palette",command=self.use_palette)
        ub.pack(side='left',expand=True,fill='x',padx=5,pady=5)
        ab=ctk.CTkButton(bf,text="Choose Another Image",command=self.pick_another)
        ab.pack(side='right',expand=True,fill='x',padx=5,pady=5)

    def use_palette(self):
        self.use_result=True
        self.destroy()

    def pick_another(self):
        self.choose_another=True
        self.destroy()

# -------------------- GUI: HSVColorPickerDialog --------------------

import colorsys

PLANE_SIZE=256

class HSVColorPickerDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("HSV Color Picker")
        self.geometry("640x400")
        self.resizable(False,False)

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.hue=0.0
        self.sat=0.0
        self.val=1.0
        self.selected_color:Optional[Tuple[int,int,int]]=None

        main_frame=ctk.CTkFrame(self)
        main_frame.pack(padx=10,pady=10,fill='both',expand=True)

        self.hue_gradient_image=self.create_hue_gradient(width=360,height=20)
        self.hue_gradient_photo=ImageTk.PhotoImage(self.hue_gradient_image)
        self.hue_gradient_label=tk.Label(main_frame,image=self.hue_gradient_photo,bd=1,relief='ridge')
        self.hue_gradient_label.grid(row=0,column=0,columnspan=2,pady=(0,5),sticky='w')

        self.hue_slider=ctk.CTkSlider(main_frame,from_=0,to=360,command=self.on_hue_changed,width=360)
        self.hue_slider.set(0)
        self.hue_slider.grid(row=1,column=0,columnspan=2,padx=0,pady=(0,10),sticky='w')

        self.plane_canvas=tk.Canvas(main_frame,width=PLANE_SIZE,height=PLANE_SIZE,bd=2,relief='sunken',cursor='cross')
        self.plane_canvas.grid(row=2,column=0,padx=(0,10),pady=5)
        self.plane_canvas.bind("<Button-1>",self.on_plane_click)
        self.plane_canvas.bind("<B1-Motion>",self.on_plane_click)
        self.plane_image=None
        self.plane_photo=None
        self.circle_id=None

        self.create_color_representations(main_frame)

        ok_button=ctk.CTkButton(self,text="OK",command=self.on_ok)
        ok_button.pack(side='bottom',pady=(0,10))

        self.update_color_plane()
        self.update_preview()

    def create_hue_gradient(self,width=360,height=20):
        img=Image.new("RGB",(width,height),"black")
        for x in range(width):
            hue_norm=x/float(width)
            h=hue_norm*360
            r,g,b=colorsys.hsv_to_rgb(h/360.0,1.0,1.0)
            for y in range(height):
                img.putpixel((x,y),(int(r*255),int(g*255),int(b*255)))
        return img

    def on_hue_changed(self,new_hue):
        self.hue=float(new_hue)
        self.update_color_plane()
        self.update_preview()
        self.update_color_reps()

    def on_plane_click(self,event):
        x=event.x
        y=event.y
        if x<0: x=0
        if x>=PLANE_SIZE: x=PLANE_SIZE-1
        if y<0: y=0
        if y>=PLANE_SIZE: y=PLANE_SIZE-1

        self.sat=x/(PLANE_SIZE-1)
        self.val=1.0-(y/(PLANE_SIZE-1))
        self.update_preview()
        self.update_circle()
        self.update_color_reps()

    def update_color_plane(self):
        img=Image.new("RGB",(PLANE_SIZE,PLANE_SIZE),"black")
        hue_norm=self.hue/360.0
        for j in range(PLANE_SIZE):
            v=1.0-j/float(PLANE_SIZE-1)
            for i in range(PLANE_SIZE):
                s=i/float(PLANE_SIZE-1)
                r,g,b=colorsys.hsv_to_rgb(hue_norm,s,v)
                img.putpixel((i,j),(int(r*255),int(g*255),int(b*255)))
        self.plane_image=img
        self.plane_photo=ImageTk.PhotoImage(img)
        self.plane_canvas.create_image(0,0,anchor='nw',image=self.plane_photo)
        self.update_circle()

    def update_preview(self):
        r,g,b=self.get_rgb()
        hx=f"#{r:02x}{g:02x}{b:02x}"
        if hasattr(self,'preview_box'):
            self.preview_box.configure(fg_color=hx)

    def update_circle(self):
        if self.plane_photo is None: return
        if self.circle_id is not None:
            self.plane_canvas.delete(self.circle_id)
            self.circle_id=None

        x=self.sat*(PLANE_SIZE-1)
        y=(1.0-self.val)*(PLANE_SIZE-1)
        rad=5
        x0=x-rad
        y0=y-rad
        x1=x+rad
        y1=y+rad
        try:
            bgc=self.plane_image.getpixel((int(x),int(y)))
            lum=0.2126*bgc[0]+0.7152*bgc[1]+0.0722*bgc[2]
            if lum>128:
                oc="#000000"
            else:
                oc="#FFFFFF"
        except:
            oc="#FFFFFF"

        self.circle_id=self.plane_canvas.create_oval(x0,y0,x1,y1,outline=oc,width=2)

    def create_color_representations(self,parent):
        rf=ctk.CTkFrame(parent)
        rf.grid(row=2,column=1,padx=10,pady=5,sticky='n')

        rgb_lab=ctk.CTkLabel(rf,text="RGB:")
        rgb_lab.grid(row=0,column=0,padx=5,pady=(0,5),sticky='w')

        self.r_var=tk.StringVar(value="255")
        self.g_var=tk.StringVar(value="255")
        self.b_var=tk.StringVar(value="255")

        self.r_entry=ctk.CTkEntry(rf,textvariable=self.r_var,width=60)
        self.g_entry=ctk.CTkEntry(rf,textvariable=self.g_var,width=60)
        self.b_entry=ctk.CTkEntry(rf,textvariable=self.b_var,width=60)

        self.r_entry.bind("<Return>",self.on_rgb_enter)
        self.g_entry.bind("<Return>",self.on_rgb_enter)
        self.b_entry.bind("<Return>",self.on_rgb_enter)

        self.r_entry.grid(row=0,column=1,padx=5,pady=(0,5))
        self.g_entry.grid(row=0,column=2,padx=5,pady=(0,5))
        self.b_entry.grid(row=0,column=3,padx=5,pady=(0,5))

        hex_lab=ctk.CTkLabel(rf,text="HEX:")
        hex_lab.grid(row=1,column=0,padx=5,pady=(10,5),sticky='w')

        self.hex_var=tk.StringVar(value="#FFFFFF")
        self.hex_entry=ctk.CTkEntry(rf,textvariable=self.hex_var,width=180)
        self.hex_entry.bind("<Return>",self.on_hex_enter)
        self.hex_entry.grid(row=1,column=1,columnspan=3,padx=(5,0),pady=(10,5),sticky='w')

        prev_lab=ctk.CTkLabel(rf,text="Selected Color:")
        prev_lab.grid(row=2,column=0,padx=5,pady=(10,5),sticky='w')

        self.preview_box=ctk.CTkLabel(rf,text="",width=80,height=40,fg_color="#ffffff",corner_radius=6)
        self.preview_box.grid(row=2,column=1,padx=5,pady=(10,5),sticky='w')

    def get_rgb(self)->Tuple[int,int,int]:
        import colorsys
        r,g,b=colorsys.hsv_to_rgb(self.hue/360.0,self.sat,self.val)
        return int(r*255),int(g*255),int(b*255)

    def update_color_reps(self):
        r,g,b=self.get_rgb()
        self.r_var.set(str(r))
        self.g_var.set(str(g))
        self.b_var.set(str(b))
        self.hex_var.set(f"#{r:02x}{g:02x}{b:02x}")

    def on_rgb_enter(self,event):
        try:
            r=int(self.r_var.get())
            g=int(self.g_var.get())
            b=int(self.b_var.get())
            if r<0 or g<0 or b<0 or r>255 or g>255 or b>255:
                raise ValueError("RGB must be [0..255]")
            import colorsys
            h,s,v=colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
            self.hue=h*360
            self.sat=s
            self.val=v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as e:
            messagebox.showerror("Invalid Input",str(e))

    def on_hex_enter(self,event):
        try:
            hx=self.hex_var.get().strip()
            if not hx.startswith('#') or len(hx)!=7:
                raise ValueError("HEX code must be #RRGGBB.")
            r=int(hx[1:3],16)
            g=int(hx[3:5],16)
            b=int(hx[5:7],16)
            import colorsys
            h,s,v=colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
            self.hue=h*360
            self.sat=s
            self.val=v
            self.hue_slider.set(self.hue)
            self.update_color_plane()
            self.update_preview()
            self.update_circle()
            self.update_color_reps()
        except Exception as e:
            messagebox.showerror("Invalid Input",str(e))

    def on_ok(self):
        self.selected_color=self.get_rgb()
        self.destroy()

# -------------------- GUI: CustomPaletteCreator --------------------

class CustomPaletteCreator(ctk.CTkToplevel):
    def __init__(self, parent, custom_palettes, save_callback, refresh_callback):
        super().__init__(parent)
        self.title("Create Custom Palette")
        self.geometry("500x400")
        self.resizable(False,False)

        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.custom_palettes=custom_palettes
        self.save_callback=save_callback
        self.refresh_callback=refresh_callback
        self.colors:List[Tuple[int,int,int]]=[]

        self.palette_frame=ctk.CTkFrame(self)
        self.palette_frame.pack(padx=10,pady=10,fill='both',expand=True)

        self.update_palette_display()

        self.save_button=ctk.CTkButton(self,text="Save Palette",command=self.save_palette)
        self.save_button.pack(pady=10)

    def update_palette_display(self):
        for w in self.palette_frame.winfo_children():
            w.destroy()

        square_size=40
        pad=5
        for idx,col in enumerate(self.colors):
            hx=f'#{col[0]:02x}{col[1]:02x}{col[2]:02x}'
            btn=tk.Button(self.palette_frame,bg=hx,width=4,height=2,relief='raised',cursor='hand2')
            btn.grid(row=idx//10,column=idx%10,padx=pad,pady=pad)
            btn.bind("<Button-3>",lambda ev,i=idx: self.delete_color(i))

        plus_btn=ctk.CTkButton(self.palette_frame,text="+",width=square_size,height=square_size,command=self.add_color,corner_radius=8)
        plus_btn.grid(row=len(self.colors)//10,column=len(self.colors)%10,padx=pad,pady=pad)

    def add_color(self):
        pick=HSVColorPickerDialog(self)
        pick.wait_window()
        if pick.selected_color is not None:
            self.colors.append(pick.selected_color)
            self.update_palette_display()

    def delete_color(self, index:int):
        if 0<=index<len(self.colors):
            del self.colors[index]
            self.update_palette_display()

    def save_palette(self):
        if not self.colors:
            messagebox.showwarning("No Colors","Please add at least one color to the palette.",parent=self)
            return
        pname=simpledialog.askstring("Palette Name","Enter a name for the custom palette:",parent=self)
        if not pname:
            return
        enames=[n for n,_ in self.custom_palettes]
        if pname in enames:
            messagebox.showerror("Duplicate Name","A palette with this name already exists.",parent=self)
            return
        self.custom_palettes.append((pname,self.colors.copy()))
        self.save_callback()
        self.refresh_callback()
        self.destroy()

# -------------------- GUI: Main App --------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Dithering Tool")
        self.geometry("1200x800")

        self.grid_rowconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=1)

        self.sidebar=ctk.CTkFrame(self,width=200)
        self.sidebar.grid(row=0,column=0,padx=10,pady=10,sticky='nsew')
        self.sidebar.grid_rowconfigure(16,weight=1)

        self.main_frame=ctk.CTkFrame(self)
        self.main_frame.grid(row=0,column=1,padx=10,pady=10,sticky='nsew')
        self.main_frame.grid_rowconfigure(0,weight=1)
        self.main_frame.grid_columnconfigure(0,weight=1)

        self.image_viewer=ZoomableImage(self.main_frame,bg="gray20",highlightthickness=0)
        self.image_viewer.grid(row=0,column=0,sticky='nsew')

        self.create_controls()

        self.current_image:Optional[Image.Image]=None
        self.original_filename:Optional[str]=None
        self.dithered_image:Optional[Image.Image]=None
        self.pixelized_image:Optional[Image.Image]=None
        self.display_state="pixelized"
        self.current_palette_name:Optional[str]=None

        self.palette_file="palette.json"
        self.custom_palettes=[]
        self.load_custom_palettes()

        self.is_video=False
        self.video_path=None
        self.last_used_palette:Optional[List[Tuple[int,int,int]]]=None

    def create_controls(self):
        row=0
        self.open_button=ctk.CTkButton(self.sidebar,text="Open Image/Video",command=self.open_image)
        self.open_button.grid(row=row,column=0,padx=20,pady=(10,5),sticky='ew')
        row+=1

        self.mode_label=ctk.CTkLabel(self.sidebar,text="Dither Mode:")
        self.mode_label.grid(row=row,column=0,padx=20,pady=(10,0),sticky='w')
        row+=1

        self.dither_mode=ctk.CTkOptionMenu(self.sidebar,values=[m.value for m in DitherMode])
        self.dither_mode.grid(row=row,column=0,padx=20,pady=(0,10),sticky='ew')
        self.dither_mode.set(DitherMode.BAYER4x4.value)
        row+=1

        self.colors_label=ctk.CTkLabel(self.sidebar,text="Number of Colors:")
        self.colors_label.grid(row=row,column=0,padx=20,pady=(10,0),sticky='w')
        row+=1

        self.colors_entry=ctk.CTkEntry(self.sidebar,placeholder_text="16")
        self.colors_entry.insert(0,"16")
        self.colors_entry.grid(row=row,column=0,padx=20,pady=(0,10),sticky='ew')
        row+=1

        self.auto_pixelize_var=tk.BooleanVar(value=True)
        self.auto_pixelize_check=ctk.CTkCheckBox(
            self.sidebar,
            text="Automatic Pixelization",
            variable=self.auto_pixelize_var,
            command=self.toggle_auto_pixelization
        )
        self.auto_pixelize_check.grid(row=row,column=0,padx=20,pady=(0,10),sticky='w')
        row+=1

        self.max_size_label=ctk.CTkLabel(self.sidebar,text="Maximum Size:")
        self.max_size_entry=ctk.CTkEntry(self.sidebar,placeholder_text="640")
        self.max_size_entry.insert(0,"640")

        self.max_size_label.grid(row=row,column=0,padx=20,pady=(0,0),sticky='w')
        self.max_size_entry.grid(row=row+1,column=0,padx=20,pady=(0,10),sticky='ew')
        self.max_size_label.grid_remove()
        self.max_size_entry.grid_remove()
        row+=2

        self.apply_button=ctk.CTkButton(self.sidebar,text="Apply Dithering",command=self.show_palette_dialog)
        self.apply_button.grid(row=row,column=0,padx=20,pady=10,sticky='ew')
        row+=1

        # --- Gamma Correction Checkbox on one row with its text ---
        self.use_gamma_var = tk.BooleanVar(value=False)
        self.gamma_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Use Gamma Correction",
            variable=self.use_gamma_var
        )
        self.gamma_check.grid(row=row,column=0,padx=20,pady=(0,10),sticky='w')
        row+=1
        # ----------------------------------------------------------

        self.save_button=ctk.CTkButton(self.sidebar,text="Save Image",command=self.save_image)
        self.save_button.grid(row=row,column=0,padx=20,pady=10,sticky='ew')
        row+=1

        self.pixelize_button=ctk.CTkButton(self.sidebar,text="Pixelize",command=self.pixelize_image)
        self.pixelize_button.grid(row=row,column=0,padx=20,pady=10,sticky='ew')
        row+=1

        self.reset_button=ctk.CTkButton(self.sidebar,text="Fit to Window",command=self.fit_to_window)
        self.reset_button.grid(row=row,column=0,padx=20,pady=10,sticky='ew')
        row+=1

        self.toggle_button=ctk.CTkButton(self.sidebar,text="Toggle View",command=self.toggle_view)
        self.toggle_button.grid(row=row,column=0,padx=20,pady=10,sticky='ew')
        row+=1

        self.random_frame_button=ctk.CTkButton(self.sidebar,text="New Random Frame",command=self.load_random_frame)
        self.random_frame_button.grid(row=row,column=0,padx=20,pady=(10,5),sticky='ew')
        row+=1
        self.random_frame_button.grid_remove()

        self.apply_video_button=ctk.CTkButton(self.sidebar,text="Apply to Video",command=self.apply_to_video)
        self.apply_video_button.grid(row=row,column=0,padx=20,pady=(10,5),sticky='ew')
        row+=1
        self.apply_video_button.grid_remove()

        # Let remaining rows fill
        for i in range(row,17):
            self.sidebar.grid_rowconfigure(i,weight=1)


    def toggle_auto_pixelization(self):
        if self.auto_pixelize_var.get():
            self.max_size_label.grid_remove()
            self.max_size_entry.grid_remove()
        else:
            self.max_size_label.grid()
            self.max_size_entry.grid()

    def open_image(self):
        fp=filedialog.askopenfilename(
            filetypes=[
                ("Image/Video files","*.png *.jpg *.jpeg *.gif *.bmp *.mp4 *.mkv *.avi *.mov"),
                ("All files","*.*")
            ]
        )
        if not fp: return
        ext=os.path.splitext(fp)[1].lower()
        vexts=[".mp4",".mkv",".avi",".mov"]
        if ext in vexts:
            self.is_video=True
            self.video_path=fp
            self.random_frame_button.grid()
            self.apply_video_button.grid()
            self.load_random_frame()
        else:
            self.is_video=False
            self.video_path=None
            self.random_frame_button.grid_remove()
            self.apply_video_button.grid_remove()
            try:
                self.current_image=Image.open(fp)
                self.original_filename=fp
            except Exception as e:
                messagebox.showerror("Error",f"Failed to open image:\n{e}")
                return
            if self.auto_pixelize_var.get():
                self.pixelize_image(auto=True)
            else:
                self.pixelized_image=self.current_image.convert("RGB")
                self.display_state="pixelized"
                self.image_viewer.set_image(self.pixelized_image)
                self.fit_to_window()
                self.dithered_image=None
                self.last_used_palette=None

    def load_random_frame(self):
        if not self.is_video or not self.video_path: return
        try:
            cmd=[
                "ffprobe","-v","error","-count_frames","-select_streams","v:0",
                "-show_entries","stream=nb_read_frames",
                "-of","default=nokey=1:noprint_wrappers=1",self.video_path
            ]
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            total_frames_str=proc.stdout.strip()
            total_frames=int(total_frames_str) if total_frames_str else 0
            if total_frames<1:
                raise ValueError("Failed to get frame count.")

            idx=random.randint(1,total_frames-1)
            tmp_frame="tmp_preview_frame.png"
            ext_cmd=[
                "ffmpeg","-y","-i",self.video_path,
                "-vf",f"select='eq(n,{idx})'",
                "-vframes","1",
                tmp_frame
            ]
            subprocess.run(ext_cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)

            self.current_image=Image.open(tmp_frame)
            self.original_filename=self.video_path
            if self.auto_pixelize_var.get():
                self.pixelize_image(auto=True)
            else:
                self.pixelized_image=self.current_image.convert('RGB')
                self.display_state="pixelized"
                self.image_viewer.set_image(self.pixelized_image)
                self.fit_to_window()
                self.dithered_image=None
                self.last_used_palette=None

            os.remove(tmp_frame)

        except Exception as e:
            messagebox.showerror("Video Error",f"Failed to load random frame:\n{e}")

    def show_palette_dialog(self):
        if not self.pixelized_image:
            messagebox.showwarning("No Pixelized Image","Please pixelize the image first.")
            return
        dlg=PaletteDialog(self,self.pixelized_image,self.custom_palettes,self.save_custom_palettes)
        self.wait_window(dlg)
        if dlg.selected_palette:
            try:
                nc=int(self.colors_entry.get())
                if nc<=0: raise ValueError
            except:
                messagebox.showerror("Invalid Number of Colors","Please enter a valid positive integer.")
                return
            try:
                dm=DitherMode(self.dither_mode.get())
            except:
                messagebox.showerror("Invalid Dither Mode","Please select a valid dither mode.")
                return

            self.current_palette_name=dlg.selected_palette_name or "UnknownPalette"
            use_gamma = self.use_gamma_var.get()
            ditherer=ImageDitherer(num_colors=nc,dither_mode=dm,palette=dlg.selected_palette,use_gamma=use_gamma)
            try:
                self.dithered_image=ditherer.apply_dithering(self.pixelized_image)
                self.last_used_palette=dlg.selected_palette
            except Exception as e:
                messagebox.showerror("Dithering Error",f"An error occurred:\n{e}")
                return
            self.display_state="dithered"
            self.image_viewer.set_image(self.dithered_image)
            self.fit_to_window()

    def pixelize_image(self, auto=False):
        if not self.current_image:
            if not auto:
                messagebox.showwarning("No Image","Please open an image (or video) first.")
            return

        if self.auto_pixelize_var.get() or auto:
            mx=640
        else:
            try:
                mx=int(self.max_size_entry.get())
                if mx<=0: raise ValueError
            except:
                messagebox.showerror("Invalid Maximum Size","Please enter a valid positive integer.")
                return

        ratio=self.current_image.width/self.current_image.height
        if self.current_image.width>=self.current_image.height:
            nw=mx
            nh=int(mx/ratio)
        else:
            nh=mx
            nw=int(mx*ratio)

        # ensure even dimension
        nw=(nw//2)*2
        nh=(nh//2)*2

        resized=self.current_image.resize((nw,nh), Image.Resampling.NEAREST)
        final=resized.convert('RGB')
        self.pixelized_image=final
        self.display_state="pixelized"
        self.image_viewer.set_image(self.pixelized_image)
        self.fit_to_window()
        self.dithered_image=None
        self.last_used_palette=None

        if not auto:
            messagebox.showinfo("Pixelization Complete","Image has been pixelized.")

    def save_image(self):
        if self.display_state=="dithered":
            image_to_save=self.dithered_image
        elif self.display_state=="pixelized":
            image_to_save=self.pixelized_image
        else:
            image_to_save=None

        if not image_to_save:
            messagebox.showwarning("No Image to Save","There is no image to save.")
            return

        if self.original_filename:
            base_name=os.path.splitext(os.path.basename(self.original_filename))[0]
        else:
            base_name="untitled"

        parts=[base_name]
        if self.display_state=="pixelized":
            parts.append("pixelized")
        elif self.display_state=="dithered":
            parts.append("dithered")
            parts.append(self.dither_mode.get())

        if self.current_palette_name:
            safe_pn=self.current_palette_name.replace(' ','_')
            parts.append(safe_pn)

        try:
            nc=int(self.colors_entry.get())
        except:
            nc=16
        parts.append(f"{nc}colors")

        # Also note gamma usage in file name if used
        if self.use_gamma_var.get():
            parts.append("gamma")

        default_filename='_'.join(parts)+".png"

        fp=filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("PNG files","*.png"),("All files","*.*")]
        )
        if fp:
            try:
                image_to_save.save(fp)
                messagebox.showinfo("Image Saved",f"Image saved as: {fp}")
            except Exception as e:
                messagebox.showerror("Save Error",f"Failed to save image:\n{e}")

    def fit_to_window(self):
        self.image_viewer.fit_to_window()

    def toggle_view(self):
        if self.display_state=="pixelized":
            if self.dithered_image:
                self.image_viewer.set_image(self.dithered_image)
                self.display_state="dithered"
            else:
                messagebox.showwarning("No Dithered Image","Please apply dithering first.")
        elif self.display_state=="dithered":
            self.image_viewer.set_image(self.pixelized_image)
            self.display_state="pixelized"
        self.fit_to_window()

    def load_custom_palettes(self):
        if not os.path.exists(self.palette_file):
            with open(self.palette_file,'w') as f:
                json.dump([],f)
            return
        try:
            with open(self.palette_file,'r') as f:
                data=json.load(f)
            for p in data:
                nm=p['name']
                def hx2rgb(hx:str)->Tuple[int,int,int]:
                    hx=hx.lstrip('#')
                    return tuple(int(hx[i:i+2],16) for i in (0,2,4))
                cols=[hx2rgb(c) for c in p['colors']]
                self.custom_palettes.append((nm,cols))
        except Exception as e:
            messagebox.showerror("Error",f"Failed to load custom palettes:\n{e}")
            self.custom_palettes=[]

    def save_custom_palettes(self):
        data=[]
        for nm,cols in self.custom_palettes:
            def rgb2hx(rgb:Tuple[int,int,int])->str:
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            hxcols=[rgb2hx(c) for c in cols]
            data.append({"name":nm,"colors":hxcols})
        try:
            with open(self.palette_file,'w') as f:
                json.dump(data,f,indent=4)
        except Exception as e:
            messagebox.showerror("Error",f"Failed to save custom palettes:\n{e}")

    def rgb_to_hex(self,rgb:Tuple[int,int,int])->str:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def hex_to_rgb(self,hx:str)->Tuple[int,int,int]:
        hx=hx.lstrip('#')
        return tuple(int(hx[i:i+2],16) for i in (0,2,4))

    def apply_to_video(self):
        if not self.is_video or not self.video_path:
            messagebox.showwarning("Not Video","No video to process.")
            return
        if self.dithered_image is None and self.pixelized_image is None:
            messagebox.showwarning("No Dither","Please dither or pixelize at least once.")
            return

        if self.original_filename:
            base_name=os.path.splitext(os.path.basename(self.original_filename))[0]
        else:
            base_name="untitled"

        parts=[base_name]
        if self.dithered_image is not None:
            parts.append("dithered")
            parts.append(self.dither_mode.get())
        else:
            parts.append("pixelized")

        if self.current_palette_name:
            sp=self.current_palette_name.replace(' ','_')
            parts.append(sp)

        try:
            nc=int(self.colors_entry.get())
        except:
            nc=16
        parts.append(f"{nc}colors")

        if self.use_gamma_var.get():
            parts.append("gamma")

        default_out='_'.join(parts)+".mp4"

        out_path=filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=default_out,
            filetypes=[("MP4 video files","*.mp4"),("All files","*.*")]
        )
        if not out_path: return

        try:
            tmp_dir="frames_tmp"
            os.makedirs(tmp_dir,exist_ok=True)

            extract_cmd=[
                "ffmpeg","-y","-i",self.video_path,
                os.path.join(tmp_dir,"frame_%05d.png")
            ]
            subprocess.run(extract_cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)

            fps_cmd=[
                "ffprobe","-v","error","-select_streams","v:0",
                "-show_entries","stream=r_frame_rate",
                "-of","default=nokey=1:noprint_wrappers=1",
                self.video_path
            ]
            fps_proc=subprocess.run(fps_cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
            raw=fps_proc.stdout.strip()
            if raw and "/" in raw:
                num,den=raw.split("/")
                try:
                    float_fps=float(num)/float(den)
                except:
                    float_fps=30.0
            else:
                float_fps=30.0

            if self.last_used_palette is not None:
                palette_for_video=self.last_used_palette
            else:
                palette_for_video=ColorReducer.reduce_colors(self.pixelized_image,nc)

            try:
                dm=DitherMode(self.dither_mode.get())
            except:
                dm=DitherMode.BAYER4x4

            use_gamma = self.use_gamma_var.get()
            dith=ImageDitherer(num_colors=nc,dither_mode=dm,palette=palette_for_video,use_gamma=use_gamma)

            frame_files=sorted(glob.glob(os.path.join(tmp_dir,"frame_*.png")))
            total_frames=len(frame_files)

            for i,fpn in enumerate(frame_files,start=1):
                # ------------------ ADDED DEBUG BLOCK ------------------
                try:
                    # Attempt reading & dithering each frame
                    frm=Image.open(fpn)
                    if not self.auto_pixelize_var.get():
                        try:
                            mxv=int(self.max_size_entry.get())
                        except:
                            mxv=640
                        ratio=frm.width/frm.height
                        if frm.width>=frm.height:
                            nw=mxv
                            nh=int(mxv/ratio)
                        else:
                            nh=mxv
                            nw=int(mxv*ratio)
                        nw=(nw//2)*2
                        nh=(nh//2)*2
                        frm=frm.resize((nw,nh), Image.Resampling.NEAREST)
                    else:
                        ratio=frm.width/frm.height
                        if frm.width>=frm.height:
                            nw=640
                            nh=int(640/ratio)
                        else:
                            nh=640
                            nw=int(640*ratio)
                        nw=(nw//2)*2
                        nh=(nh//2)*2
                        frm=frm.resize((nw,nh), Image.Resampling.NEAREST)

                    dimg=dith.apply_dithering(frm)
                    dimg.save(fpn)
                except Exception as e:
                    # Print out an immediate debug message
                    print(f"\nERROR on frame {i} / {fpn}: {e}", file=sys.stderr)
                    # optionally remove or skip that frame:
                    # os.remove(fpn)
                    continue
                # -------------------------------------------------------

                prog=float(i)/total_frames
                bar_len=30
                filled=int(bar_len*prog)
                bar='#'*filled + '-'*(bar_len-filled)
                sys.stdout.write(f"\rDithering frames: [{bar}] {i}/{total_frames}")
                sys.stdout.flush()

            print()

            encode_cmd=[
                "ffmpeg","-y",
                "-framerate",f"{float_fps:.5f}",
                "-i",os.path.join(tmp_dir,"frame_%05d.png"),
                "-i",self.video_path,
                "-map","0:v",
                "-map","1:a?",
                "-map","1:s?",
                "-c:v","libx264",
                "-pix_fmt","yuv420p",
                "-c:a","copy",
                "-c:s","copy",
                "-r",f"{float_fps:.5f}",
                out_path
            ]
            subprocess.run(encode_cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)

            shutil.rmtree(tmp_dir)
            messagebox.showinfo("Video Complete",f"Video successfully dithered and saved to: {out_path}")

        except Exception as e:
            messagebox.showerror("Video Processing Error",f"Failed to process video:\n{e}")


# -------------------- CLI Tool --------------------

def run_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Image Dithering Tool CLI",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command',help='Commands')

    pixelize_parser = subparsers.add_parser('pixelize',help='Pixelize an image/video.')
    pixelize_parser.add_argument('input_image',type=str,help='Path to input image/video.')
    pixelize_parser.add_argument('output_image',type=str,help='Path to save pixelized result.')
    pixelize_parser.add_argument('-m','--max-size',type=int,default=640,help='Max dimension for pixelization.')

    dither_parser = subparsers.add_parser('dither',help='Apply dithering to an image/video.')
    dither_parser.add_argument('input_image',type=str,help='Path to input.')
    dither_parser.add_argument('output_image',type=str,help='Path to output.')
    dither_parser.add_argument('-d','--mode',choices=[m.value for m in DitherMode],default='bayer4x4',help='Dither mode.')
    dither_parser.add_argument('-c','--colors',type=int,default=16,help='Number of colors.')
    dither_parser.add_argument('--algo-palette',type=str,choices=["median_cut","kmeans_variant1","kmeans_variant2","uniform"],default=None,
                               help='Algorithmic palette for dithering.')
    dither_parser.add_argument('-p','--palette',type=str,default=None,help='Name of custom palette (from palette.json).')
    # NEW CLI arg
    dither_parser.add_argument('--gamma-correction',action='store_true',
                               help='Use gamma correction (sRGB->linear->sRGB).')

    dp_parser = subparsers.add_parser('dither-pixelize',help='Pixelize then dither an image/video.')
    dp_parser.add_argument('input_image',type=str,help='Path to input.')
    dp_parser.add_argument('output_image',type=str,help='Path to output.')
    dp_parser.add_argument('-d','--mode',choices=[m.value for m in DitherMode],default='bayer4x4',help='Dither mode.')
    dp_parser.add_argument('-c','--colors',type=int,default=16,help='Number of colors.')
    dp_parser.add_argument('--algo-palette',type=str,choices=["median_cut","kmeans_variant1","kmeans_variant2","uniform"],default=None,
                           help='Algorithmic palette.')
    dp_parser.add_argument('-p','--palette',type=str,default=None,help='Name of custom palette (from palette.json).')
    dp_parser.add_argument('-m','--max-size',type=int,default=640,help='Max dimension for pixelization.')
    # NEW CLI arg
    dp_parser.add_argument('--gamma-correction',action='store_true',
                           help='Use gamma correction (sRGB->linear->sRGB).')

    import_lospal_parser = subparsers.add_parser('import-lospal',help='Import a palette from lospec.com URL into palette.json')
    import_lospal_parser.add_argument('url',type=str,help='Full URL from lospec.com/palette-list/...')

    create_pal_img_parser = subparsers.add_parser('create-pal-from-image',help='Create a palette from an image (K-means) and store it in palette.json')
    create_pal_img_parser.add_argument('input_image',type=str,help='Path to the image from which to create a palette.')
    create_pal_img_parser.add_argument('-c','--colors',type=int,default=16,help='Number of colors for the generated palette.')
    create_pal_img_parser.add_argument('-n','--name',type=str,default=None,help='Name to store in palette.json (if omitted, we use "FromImage").')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    def hex_to_rgb(hex_code:str)->Tuple[int,int,int]:
        hex_code=hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2],16) for i in (0,2,4))

    def rgb_to_hex(rgb:Tuple[int,int,int])->str:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def load_custom_palettes(filename="palette.json")->List[Tuple[str,List[Tuple[int,int,int]]]]:
        pals=[]
        if os.path.exists(filename):
            try:
                with open(filename,'r') as f:
                    data=json.load(f)
                for p in data:
                    nm=p['name']
                    cl=[hex_to_rgb(c) for c in p['colors']]
                    pals.append((nm,cl))
            except Exception as e:
                print(f"Warning: fail load palettes: {e}",file=sys.stderr)
        else:
            with open(filename,'w') as f:
                json.dump([],f)
        return pals

    def save_custom_palettes(filename:str, pal_list:List[Tuple[str,List[Tuple[int,int,int]]]]):
        data=[]
        for nm,cols in pal_list:
            hxcols=[rgb_to_hex(c) for c in cols]
            data.append({"name":nm,"colors":hxcols})
        with open(filename,'w') as f:
            json.dump(data,f,indent=4)

    custom_palettes=load_custom_palettes("palette.json")

    def is_video_file(path:str)->bool:
        e=os.path.splitext(path)[1].lower()
        return e in [".mp4",".mkv",".avi",".mov"]

    def pixelize_image_cli(img:Image.Image, max_size:int)->Image.Image:
        ratio=img.width/img.height
        if img.width>=img.height:
            nw=max_size
            nh=int(max_size/ratio)
        else:
            nh=max_size
            nw=int(max_size*ratio)
        nw=(nw//2)*2
        nh=(nh//2)*2
        resized=img.resize((nw,nh),Image.Resampling.NEAREST)
        return resized.convert('RGB')

    def generate_kmeans_palette_cli(img:Image.Image,num_colors:int,random_state=42)->List[Tuple[int,int,int]]:
        arr=np.array(img.convert('RGB'))
        pix=arr.reshape(-1,3)
        if len(pix)>10000:
            idx=np.random.choice(len(pix),10000,replace=False)
            pix=pix[idx]
        km=KMeans(n_clusters=num_colors,random_state=random_state)
        km.fit(pix)
        c=km.cluster_centers_.astype(int)
        return [tuple(x) for x in c]

    def generate_uniform_palette_cli(num_colors:int)->List[Tuple[int,int,int]]:
        c=[]
        cube=int(math.ceil(num_colors**(1/3)))
        for r in range(cube):
            for g in range(cube):
                for b in range(cube):
                    if len(c)>=num_colors:
                        break
                    rr=int(r*255/(cube-1)) if cube>1 else 128
                    gg=int(g*255/(cube-1)) if cube>1 else 128
                    bb=int(b*255/(cube-1)) if cube>1 else 128
                    c.append((rr,gg,bb))
        return c[:num_colors]

    def generate_median_cut_palette_cli(img:Image.Image,num_colors:int)->List[Tuple[int,int,int]]:
        return ColorReducer.reduce_colors(img,num_colors)

    def get_algorithmic_palette_cli(img:Image.Image,algo:str,num_colors:int)->List[Tuple[int,int,int]]:
        if algo=="median_cut":
            return generate_median_cut_palette_cli(img,num_colors)
        elif algo=="kmeans_variant1":
            return generate_kmeans_palette_cli(img,num_colors,random_state=42)
        elif algo=="kmeans_variant2":
            return generate_kmeans_palette_cli(img,num_colors,random_state=123)
        elif algo=="uniform":
            return generate_uniform_palette_cli(num_colors)
        else:
            return generate_median_cut_palette_cli(img,num_colors)

    def do_pixel_dither(input_path:str,output_path:str, do_pixel:bool, do_dither:bool,
                        mode:str, colors:int, algo_palette:Optional[str],
                        pal_name:Optional[str], max_size:int, use_gamma:bool=False):
        used_palette=None
        if pal_name:
            found=None
            for n,c in custom_palettes:
                if n.lower()==pal_name.lower():
                    found=c
                    break
            if not found:
                print(f"Error: custom palette '{pal_name}' not found.",file=sys.stderr)
                sys.exit(1)
            used_palette=found

        if is_video_file(input_path):
            tmp_dir="cli_frames_tmp"
            os.makedirs(tmp_dir,exist_ok=True)

            try:
                subprocess.run(["ffmpeg","-y","-i",input_path,os.path.join(tmp_dir,"frame_%05d.png")],check=True)

                p=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                                  "-show_entries","stream=r_frame_rate",
                                  "-of","default=nokey=1:noprint_wrappers=1",input_path],
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
                raw=p.stdout.strip()
                if raw and "/" in raw:
                    n,d=raw.split("/")
                    try:
                        fps=float(n)/float(d)
                    except:
                        fps=30.0
                else:
                    fps=30.0

                frames=sorted(glob.glob(os.path.join(tmp_dir,"frame_*.png")))
                if not frames:
                    print("No frames extracted.",file=sys.stderr)
                    shutil.rmtree(tmp_dir)
                    sys.exit(1)

                # If no custom palette provided, generate from first frame if needed
                if used_palette is None and do_dither:
                    test_img=Image.open(frames[0])
                    if do_pixel:
                        test_img=pixelize_image_cli(test_img,max_size)
                    if algo_palette:
                        used_palette=get_algorithmic_palette_cli(test_img,algo_palette,colors)
                    else:
                        used_palette=generate_median_cut_palette_cli(test_img,colors)

                try:
                    dmode=DitherMode(mode)
                except:
                    dmode=DitherMode.BAYER4x4

                dith=ImageDitherer(colors,dmode,used_palette,use_gamma=use_gamma)

                totalf=len(frames)
                for i,fpth in enumerate(frames,start=1):
                    frm=Image.open(fpth)
                    if do_pixel:
                        frm=pixelize_image_cli(frm,max_size)
                    if do_dither:
                        frm=dith.apply_dithering(frm)
                    frm.save(fpth)

                    prog=i/totalf
                    bar_len=30
                    fill=int(bar_len*prog)
                    bar='#'*fill + '-'*(bar_len-fill)
                    sys.stdout.write(f"\rProcessing: [{bar}] {i}/{totalf}")
                    sys.stdout.flush()

                print()

                # Re-encode video
                subprocess.run([
                    "ffmpeg","-y",
                    "-framerate",f"{fps:.5f}",
                    "-i",os.path.join(tmp_dir,"frame_%05d.png"),
                    "-i",input_path,
                    "-map","0:v","-map","1:a?","-map","1:s?",
                    "-c:v","libx264","-pix_fmt","yuv420p",
                    "-c:a","copy","-c:s","copy",
                    "-r",f"{fps:.5f}",
                    output_path
                ],check=True)

                shutil.rmtree(tmp_dir)
                print(f"Video processed and saved to {output_path}")
            except Exception as e:
                print(f"Error: {e}",file=sys.stderr)
                shutil.rmtree(tmp_dir)
                sys.exit(1)
        else:
            # single image
            try:
                img=Image.open(input_path)
            except Exception as e:
                print(f"Error loading image: {e}",file=sys.stderr)
                sys.exit(1)
            if do_pixel:
                img=pixelize_image_cli(img,max_size)

            if used_palette is None and do_dither:
                if algo_palette:
                    used_palette=get_algorithmic_palette_cli(img,algo_palette,colors)
                else:
                    used_palette=generate_median_cut_palette_cli(img,colors)

            try:
                dmode=DitherMode(mode)
            except:
                dmode=DitherMode.BAYER4x4

            dith=ImageDitherer(colors,dmode,used_palette,use_gamma=use_gamma)
            result= dith.apply_dithering(img) if do_dither else img
            try:
                result.save(output_path)
                print(f"Image processed and saved to {output_path}")
            except Exception as e:
                print(f"Error saving image: {e}",file=sys.stderr)
                sys.exit(1)

    cmd=args.command
    if cmd=='pixelize':
        do_pixel_dither(args.input_image,args.output_image,
                        do_pixel=True,do_dither=False,
                        mode='none',colors=16,
                        algo_palette=None,pal_name=None,
                        max_size=args.max_size,
                        use_gamma=False)

    elif cmd=='dither':
        if args.algo_palette and args.palette:
            print("Error: Choose either --algo-palette or --palette, not both.",file=sys.stderr)
            sys.exit(1)
        do_pixel_dither(args.input_image,args.output_image,
                        do_pixel=False,do_dither=True,
                        mode=args.mode,colors=args.colors,
                        algo_palette=args.algo_palette,
                        pal_name=args.palette,
                        max_size=640,
                        use_gamma=args.gamma_correction)

    elif cmd=='dither-pixelize':
        if args.algo_palette and args.palette:
            print("Error: Choose either --algo-palette or --palette, not both.",file=sys.stderr)
            sys.exit(1)
        do_pixel_dither(args.input_image,args.output_image,
                        do_pixel=True,do_dither=True,
                        mode=args.mode,colors=args.colors,
                        algo_palette=args.algo_palette,
                        pal_name=args.palette,
                        max_size=args.max_size,
                        use_gamma=args.gamma_correction)

    elif cmd=='import-lospal':
        url=args.url
        def save_pals():
            save_custom_palettes("palette.json",custom_palettes)

        try:
            import urllib.request
            data=None
            with urllib.request.urlopen(url) as resp:
                data=resp.read()
            if not data:
                print("Error: Could not retrieve palette JSON.",file=sys.stderr)
                sys.exit(1)
            pjson=json.loads(data)
            name=pjson['name']
            cols=pjson['colors']
            def hx_to_rgb(hx:str)->Tuple[int,int,int]:
                hx=hx.lstrip('#')
                return tuple(int(hx[i:i+2],16) for i in (0,2,4))
            rgbc=[hx_to_rgb(f"#{c}") for c in cols]
            existing_names=[n for n,_ in custom_palettes]
            if name in existing_names:
                print(f"Error: A palette named '{name}' already exists in palette.json",file=sys.stderr)
                sys.exit(1)
            custom_palettes.append((name,rgbc))
            save_pals()
            print(f"Palette '{name}' imported successfully from: {url}")
        except Exception as e:
            print(f"Error importing from lospec.com:\n{e}",file=sys.stderr)
            sys.exit(1)

    elif cmd=='create-pal-from-image':
        input_image=args.input_image
        ccount=args.colors
        pname=args.name
        if not pname:
            pname="FromImage"

        def save_pals():
            save_custom_palettes("palette.json",custom_palettes)

        try:
            new_img=Image.open(input_image)
        except Exception as e:
            print(f"Error loading image: {e}",file=sys.stderr)
            sys.exit(1)

        arr_full=np.array(new_img.convert('RGB'))
        allpix=arr_full.reshape(-1,3)
        unique_pix=np.unique(allpix,axis=0)
        unique_count=unique_pix.shape[0]
        if unique_count<ccount:
            final_count=unique_count
        else:
            final_count=ccount
        if final_count<1: final_count=1

        if len(allpix)>10000:
            idx=np.random.choice(len(allpix),10000,replace=False)
            small=allpix[idx]
        else:
            small=allpix

        km=KMeans(n_clusters=final_count,random_state=42)
        km.fit(small)
        centers=km.cluster_centers_.astype(int)
        kpal=[tuple(c) for c in centers]

        existing_names=[n for n,_ in custom_palettes]
        if pname in existing_names:
            print(f"Error: A palette named '{pname}' already exists in palette.json",file=sys.stderr)
            sys.exit(1)

        custom_palettes.append((pname,kpal))
        save_pals()
        print(f"Created a {final_count}-color palette from: {input_image}\nStored as '{pname}' in palette.json")

    else:
        parser.print_help()
        sys.exit(1)

def main():
    if len(sys.argv)>1:
        run_cli()
    else:
        app=App()
        app.mainloop()

if __name__=="__main__":
    main()
