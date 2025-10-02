#!/usr/bin/env python3
"""
Image and Video Dithering Tool - Refactored for Performance

Key improvements:
- Multiprocessing for fast video processing
- Model caching (no reload per frame)
- Threading to prevent GUI freezing
- Modular, maintainable code
"""

import sys
import os
import threading
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image

# Local imports
from dithering_lib import (
    DitherMode,
    ImageDitherer,
    ColorReducer
)
from video_processor import VideoProcessor, NeuralPixelizer, pixelize_regular
from gui_components import (
    ZoomableImage,
    ProgressDialog,
    StatusBar
)
from utils import (
    PaletteManager,
    validate_video_file,
    validate_image_file
)

# Set appearance
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class DitheringApp(ctk.CTk):
    """
    Main GUI application with threading to prevent freezing.
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("Image Dithering Tool")
        self.geometry("1400x900")
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # State variables
        self.current_image = None
        self.pixelized_image = None
        self.dithered_image = None
        self.video_path = None
        self.is_video = False
        self.last_palette = None
        self.last_pixelization_method = "regular"
        self.display_state = "current"  # Track what we're showing: "current", "pixelized", "dithered"
        self.original_size = None  # Store original image/video size (width, height)
        
        # Palette manager
        self.palette_mgr = PaletteManager()
        
        # Video processor
        self.video_processor = VideoProcessor(progress_callback=self._on_video_progress)
        
        # Neural pixelizer (singleton, cached)
        self.neural_pix = None
        
        # Build UI
        self._create_sidebar()
        self._create_main_area()
        self._create_status_bar()
    
    def _create_sidebar(self):
        """Create the control sidebar with scrollable content."""
        # Create outer frame to hold the scrollable frame
        self.sidebar_frame = ctk.CTkFrame(self)
        self.sidebar_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.sidebar_frame.grid_rowconfigure(0, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable frame for all controls
        # Let it size based on content, no fixed width
        self.sidebar = ctk.CTkScrollableFrame(self.sidebar_frame)
        self.sidebar.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        row = 0
        
        # Title
        title_label = ctk.CTkLabel(
            self.sidebar,
            text="Dithering Tool",
            font=("Arial", 20, "bold")
        )
        title_label.grid(row=row, column=0, pady=(10, 20), padx=10)
        row += 1
        
        # Load buttons
        btn_load_img = ctk.CTkButton(
            self.sidebar,
            text="Load Image",
            command=self.load_image
        )
        btn_load_img.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        btn_load_video = ctk.CTkButton(
            self.sidebar,
            text="Load Video",
            command=self.load_video
        )
        btn_load_video.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        # Separator
        ctk.CTkLabel(self.sidebar, text="Pixelization", font=("Arial", 14, "bold")).grid(
            row=row, column=0, pady=(15, 5), padx=10
        )
        row += 1
        
        # Max size
        ctk.CTkLabel(self.sidebar, text="Max Size:").grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        self.max_size_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.max_size_entry.insert(0, "640")
        self.max_size_entry.grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        # Pixelize buttons
        btn_pixelize = ctk.CTkButton(
            self.sidebar,
            text="Pixelize (Regular)",
            command=self._on_pixelize_regular
        )
        btn_pixelize.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        btn_pixelize_neural = ctk.CTkButton(
            self.sidebar,
            text="Pixelize (Neural)",
            command=self._on_pixelize_neural
        )
        btn_pixelize_neural.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        # Separator
        ctk.CTkLabel(self.sidebar, text="Final Resize", font=("Arial", 14, "bold")).grid(
            row=row, column=0, pady=(15, 5), padx=10
        )
        row += 1
        
        # Final resize checkbox and size entry
        self.final_resize_var = tk.BooleanVar(value=False)
        final_resize_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Upscale to original size",
            variable=self.final_resize_var
        )
        final_resize_check.grid(row=row, column=0, pady=5, padx=10, sticky='w')
        row += 1
        
        ctk.CTkLabel(self.sidebar, text="Target Size (larger dim):").grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        self.final_size_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.final_size_entry.insert(0, "1920")
        self.final_size_entry.grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        # Separator
        ctk.CTkLabel(self.sidebar, text="Dithering", font=("Arial", 14, "bold")).grid(
            row=row, column=0, pady=(15, 5), padx=10
        )
        row += 1
        
        # Dither mode
        ctk.CTkLabel(self.sidebar, text="Dither Mode:").grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        self.dither_mode = ctk.StringVar(value="bayer4x4")
        dither_modes = [mode.value for mode in DitherMode]
        self.dither_dropdown = ctk.CTkOptionMenu(
            self.sidebar,
            variable=self.dither_mode,
            values=dither_modes
        )
        self.dither_dropdown.grid(row=row, column=0, pady=2, padx=10, sticky='ew')
        row += 1
        
        # Number of colors
        ctk.CTkLabel(self.sidebar, text="Colors:").grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        self.colors_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.colors_entry.insert(0, "16")
        self.colors_entry.grid(row=row, column=0, pady=2, padx=10, sticky='w')
        row += 1
        
        # Gamma correction
        self.gamma_var = tk.BooleanVar(value=False)
        gamma_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Gamma Correction",
            variable=self.gamma_var
        )
        gamma_check.grid(row=row, column=0, pady=5, padx=10, sticky='w')
        row += 1
        
        # Dither button
        btn_dither = ctk.CTkButton(
            self.sidebar,
            text="Apply Dithering",
            command=self._on_apply_dithering
        )
        btn_dither.grid(row=row, column=0, pady=10, padx=10, sticky='ew')
        row += 1
        
        # Apply to Video button (hidden by default)
        self.apply_video_button = ctk.CTkButton(
            self.sidebar,
            text="Apply to Video",
            command=self._apply_to_video_workflow
        )
        self.apply_video_button.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        self.apply_video_button.grid_remove()  # Hidden by default
        row += 1
        
        # Save button
        btn_save = ctk.CTkButton(
            self.sidebar,
            text="Save Result",
            command=self.save_image
        )
        btn_save.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        # Fit to window button
        btn_fit = ctk.CTkButton(
            self.sidebar,
            text="Fit to Window",
            command=self.fit_to_window
        )
        btn_fit.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
        
        # Toggle view button
        btn_toggle = ctk.CTkButton(
            self.sidebar,
            text="Toggle View",
            command=self.toggle_view
        )
        btn_toggle.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        row += 1
    
    def _create_main_area(self):
        """Create the main image display area."""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.image_viewer = ZoomableImage(
            self.main_frame,
            bg="gray20",
            highlightthickness=0
        )
        self.image_viewer.grid(row=0, column=0, sticky='nsew')
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=(0, 10))
    
    def load_image(self):
        """Load an image file."""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            self.current_image = Image.open(filepath).convert('RGB')
            self.pixelized_image = None
            self.dithered_image = None
            self.is_video = False
            self.video_path = None
            self.display_state = "current"
            
            # Store original size and update final size entry
            self.original_size = self.current_image.size  # (width, height)
            larger_dim = max(self.original_size)
            self.final_size_entry.delete(0, tk.END)
            self.final_size_entry.insert(0, str(larger_dim))
            
            # Hide video button
            self.apply_video_button.grid_remove()
            
            self.image_viewer.set_image(self.current_image)
            self.fit_to_window()
            self.status_bar.set_status(f"Loaded image: {Path(filepath).name} - {self.original_size[0]}x{self.original_size[1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def load_video(self):
        """Load a video file (first frame for preview)."""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Extract first frame for preview
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_path = os.path.join(tmpdir, "frame.png")
                cmd = [
                    "ffmpeg", "-i", filepath,
                    "-vframes", "1",
                    frame_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                self.current_image = Image.open(frame_path).convert('RGB')
            
            self.pixelized_image = None
            self.dithered_image = None
            self.is_video = True
            self.video_path = filepath
            self.display_state = "current"
            
            # Store original size and update final size entry
            self.original_size = self.current_image.size  # (width, height)
            larger_dim = max(self.original_size)
            self.final_size_entry.delete(0, tk.END)
            self.final_size_entry.insert(0, str(larger_dim))
            
            # Show video button
            self.apply_video_button.grid()
            
            self.image_viewer.set_image(self.current_image)
            self.fit_to_window()
            self.status_bar.set_status(f"Loaded video: {Path(filepath).name} - {self.original_size[0]}x{self.original_size[1]} (showing first frame)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{e}")
    
    def _on_pixelize_regular(self):
        """Handle regular pixelization."""
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image or video first.")
            return
        
        try:
            max_size = int(self.max_size_entry.get())
            if max_size <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for max size.")
            return
        
        self.status_bar.set_status("Pixelizing...")
        
        # Use threading to prevent GUI freeze
        def process():
            self.pixelized_image = pixelize_regular(self.current_image, max_size)
            self.last_pixelization_method = "regular"
            self.dithered_image = None
            self.display_state = "pixelized"
            
            # Update GUI from main thread
            self.after(0, lambda: self.image_viewer.set_image(self.pixelized_image))
            self.after(0, lambda: self.fit_to_window())
            self.after(0, lambda: self.status_bar.set_status("Pixelization complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _on_pixelize_neural(self):
        """Handle neural pixelization."""
        if not self.current_image:
            messagebox.showwarning("No Image", "Please load an image or video first.")
            return
        
        try:
            max_size = int(self.max_size_entry.get())
            if max_size <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for max size.")
            return
        
        self.status_bar.set_status("Neural pixelizing (this may take a moment)...")
        
        def process():
            if self.neural_pix is None:
                self.neural_pix = NeuralPixelizer()
            
            self.pixelized_image = self.neural_pix.pixelize(self.current_image, max_size)
            self.last_pixelization_method = "neural"
            self.dithered_image = None
            self.display_state = "pixelized"
            
            self.after(0, lambda: self.image_viewer.set_image(self.pixelized_image))
            self.after(0, lambda: self.fit_to_window())
            self.after(0, lambda: self.status_bar.set_status("Neural pixelization complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _on_apply_dithering(self):
        """Handle dithering - shows palette selection dialog and applies to current frame."""
        if not self.pixelized_image:
            messagebox.showwarning("No Image", "Please pixelize an image first.")
            return
        
        # Show palette dialog and apply to current frame (works for both images and video preview)
        self._show_palette_dialog_and_apply()
        
        # If video, show the "Apply to Video" button after dithering preview
        if self.is_video and self.dithered_image:
            self.apply_video_button.grid()
            self.status_bar.set_status("Preview ready. Click 'Apply to Video' to process full video.")
    
    def _show_palette_dialog_and_apply(self):
        """Show palette dialog and apply dithering to current image."""
        # Get number of colors
        try:
            num_colors = int(self.colors_entry.get())
            if num_colors <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for colors.")
            return
        
        # Show palette selection dialog
        from gui_components import PalettePreview
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Palette Method")
        dialog.geometry("450x400")
        dialog.transient(self)
        dialog.grab_set()
        
        selected_palette = [None]  # Use list to allow modification in nested function
        
        ctk.CTkLabel(dialog, text="Choose Palette Generation Method:", 
                    font=("Arial", 14, "bold")).pack(pady=10)
        
        # Generate palette options
        palette_options = []
        
        # Median Cut
        mc_palette = ColorReducer.reduce_colors(self.pixelized_image, num_colors)
        palette_options.append(("Median Cut", mc_palette))
        
        # K-means
        km_palette = ColorReducer.generate_kmeans_palette(self.pixelized_image, num_colors, random_state=42)
        palette_options.append(("K-means", km_palette))
        
        # Uniform
        uniform_palette = ColorReducer.generate_uniform_palette(num_colors)
        palette_options.append(("Uniform", uniform_palette))
        
        # Load custom palettes from palette.json
        custom_palettes_data = self.palette_mgr.palettes
        for palette_data in custom_palettes_data:
            name = palette_data['name']
            hex_colors = palette_data['colors']
            rgb_colors = [self.palette_mgr._hex_to_rgb(c) for c in hex_colors]
            palette_options.append((name, rgb_colors))
        
        # Radio buttons for selection
        selected_var = tk.StringVar(value="Median Cut")
        
        scroll_frame = ctk.CTkScrollableFrame(dialog, height=200)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        for name, palette in palette_options:
            frame = ctk.CTkFrame(scroll_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            radio = ctk.CTkRadioButton(frame, text=name, variable=selected_var, value=name)
            radio.pack(side="left", padx=5)
            
            # Show palette preview
            preview = PalettePreview(frame, palette, width=220, height=20)
            preview.pack(side="right", padx=5)
        
        def on_ok():
            # Find selected palette
            sel_name = selected_var.get()
            for name, palette in palette_options:
                if name == sel_name:
                    selected_palette[0] = palette
                    break
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        btn_frame = ctk.CTkFrame(dialog)
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="OK", command=on_ok).pack(side="right", padx=5)
        
        # Wait for dialog to close
        self.wait_window(dialog)
        
        if selected_palette[0] is None:
            return  # User cancelled
        
        # Apply dithering with selected palette
        try:
            dither_mode = DitherMode(self.dither_mode.get())
        except:
            dither_mode = DitherMode.BAYER4x4
        
        self.status_bar.set_status("Dithering...")
        self.last_palette = selected_palette[0]
        
        def process():
            ditherer = ImageDitherer(
                num_colors=num_colors,
                dither_mode=dither_mode,
                palette=selected_palette[0],
                use_gamma=self.gamma_var.get()
            )
            
            self.dithered_image = ditherer.apply_dithering(self.pixelized_image)
            
            # Apply final resize if enabled
            self.dithered_image = self._apply_final_resize(self.dithered_image)
            self.display_state = "dithered"
            
            self.after(0, lambda: self.image_viewer.set_image(self.dithered_image))
            self.after(0, lambda: self.fit_to_window())
            self.after(0, lambda: self.status_bar.set_status("Dithering complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _apply_to_video_workflow(self):
        """Handle the full video processing workflow - shows save dialog and processes."""
        if not self.last_palette or not self.dithered_image:
            messagebox.showwarning("No Palette", 
                                 "Please apply dithering to preview frame first to select a palette.")
            return
        
        # Show save dialog and process video
        self._apply_to_video()
    
    def _apply_to_video(self):
        """Apply processing to video."""
        if not self.video_path:
            return
        
        # Generate default filename for video
        default_video_name = self._generate_video_filename()
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=default_video_name,
            filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Get parameters
        try:
            num_colors = int(self.colors_entry.get())
            max_size = int(self.max_size_entry.get())
            dither_mode = DitherMode(self.dither_mode.get())
        except:
            messagebox.showerror("Invalid Input", "Please check your input values.")
            return
        
        self.status_bar.set_status("Processing video (check console for progress)...")
        
        def process():
            # Create ditherer
            ditherer = ImageDitherer(
                num_colors=num_colors,
                dither_mode=dither_mode,
                palette=self.last_palette,
                use_gamma=self.gamma_var.get()
            )
            
            # Create pixelize function if pixelized image exists
            # Note: Can't use lambdas here due to multiprocessing pickle issues
            pixelize_func = None
            if self.pixelized_image:
                if self.last_pixelization_method == "neural":
                    # Neural uses single-process mode (slower but prevents memory issues)
                    pixelize_func = ("neural", max_size)
                else:
                    # Regular uses multi-process mode (fast)
                    pixelize_func = ("regular", max_size)
            
            # Get final resize parameters
            final_resize_enabled = self.final_resize_var.get()
            final_target_size = None
            if final_resize_enabled:
                try:
                    final_target_size = int(self.final_size_entry.get())
                except:
                    final_target_size = None
            
            # Process video
            success = self.video_processor.process_video_streaming(
                self.video_path,
                output_path,
                ditherer,
                pixelize_func=pixelize_func,
                final_resize_target=final_target_size
            )
            
            # Show result
            if success:
                self.after(0, lambda: self.status_bar.set_status("Video processing complete!"))
                self.after(0, lambda: messagebox.showinfo(
                    "Success",
                    f"Video processed successfully:\n{output_path}"
                ))
            else:
                self.after(0, lambda: self.status_bar.set_status("Video processing failed"))
                self.after(0, lambda: messagebox.showerror(
                    "Error",
                    "Video processing failed. Check console for details."
                ))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _on_video_progress(self, fraction: float, message: str):
        """Callback for video processing progress."""
        # Update status bar from main thread
        progress_msg = f"{int(fraction * 100)}% - {message}"
        self.after(0, lambda: self.status_bar.set_status(progress_msg))
        # Also print to console
        print(f"Progress: {int(fraction * 100)}% - {message}")
    
    def save_image(self):
        """Save the current result."""
        if self.dithered_image:
            img_to_save = self.dithered_image
            state = "dithered"
        elif self.pixelized_image:
            img_to_save = self.pixelized_image
            state = "pixelized"
        elif self.current_image:
            img_to_save = self.current_image
            state = "original"
        else:
            messagebox.showwarning("No Image", "No image to save.")
            return
        
        # Generate default filename
        default_filename = self._generate_filename(state)
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                img_to_save.save(filepath)
                self.status_bar.set_status(f"Saved: {Path(filepath).name}")
                messagebox.showinfo("Success", f"Image saved:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def fit_to_window(self):
        """Fit the current image to window."""
        self.image_viewer.fit_to_window()
    
    def toggle_view(self):
        """Toggle between current, pixelized, and dithered views."""
        if self.display_state == "current":
            if self.pixelized_image:
                self.image_viewer.set_image(self.pixelized_image)
                self.display_state = "pixelized"
                self.fit_to_window()
            else:
                messagebox.showinfo("No Pixelized Image", "Please pixelize the image first.")
        elif self.display_state == "pixelized":
            if self.dithered_image:
                self.image_viewer.set_image(self.dithered_image)
                self.display_state = "dithered"
                self.fit_to_window()
            else:
                messagebox.showinfo("No Dithered Image", "Please apply dithering first.")
        elif self.display_state == "dithered":
            self.image_viewer.set_image(self.pixelized_image)
            self.display_state = "pixelized"
            self.fit_to_window()
    
    def _apply_final_resize(self, image: Image.Image) -> Image.Image:
        """
        Apply final resize to original size using nearest-neighbor interpolation.
        This preserves the pixelated look when upscaling.
        Ensures dimensions are even for video codec compatibility.
        
        Args:
            image: The processed image to resize
            
        Returns:
            Resized image or original if resize disabled
        """
        if not self.final_resize_var.get():
            return image
        
        try:
            target_size = int(self.final_size_entry.get())
            if target_size <= 0:
                return image
        except:
            return image
        
        current_w, current_h = image.size
        
        # Calculate new dimensions maintaining aspect ratio
        if current_w >= current_h:
            # Landscape: target_size is for width
            new_w = target_size
            new_h = int(round((current_h / current_w) * target_size))
        else:
            # Portrait: target_size is for height
            new_h = target_size
            new_w = int(round((current_w / current_h) * target_size))
        
        # Ensure dimensions are even (required for libx264 yuv420p when saving as video)
        if new_w % 2 != 0:
            new_w += 1
        if new_h % 2 != 0:
            new_h += 1
        
        # Use NEAREST to preserve pixelated look
        resized = image.resize((new_w, new_h), Image.Resampling.NEAREST)
        return resized
    
    def _generate_filename(self, state: str) -> str:
        """Generate a default filename for saving."""
        # Get base name from current file
        if self.video_path and self.is_video:
            base = Path(self.video_path).stem
        elif self.current_image:
            base = "image"
        else:
            base = "untitled"
        
        parts = [base]
        
        # Add state
        if state == "pixelized":
            parts.append("pixelized")
        elif state == "dithered":
            parts.append("dithered")
            parts.append(self.dither_mode.get())
        
        # Add color count
        try:
            colors = int(self.colors_entry.get())
            parts.append(f"{colors}colors")
        except:
            pass
        
        # Add gamma if enabled
        if self.gamma_var.get():
            parts.append("gamma")
        
        return "_".join(parts) + ".png"
    
    def _generate_video_filename(self) -> str:
        """Generate a default filename for saving video."""
        if self.video_path:
            base = Path(self.video_path).stem
        else:
            base = "video"
        
        parts = [base]
        
        # Add processing info
        if self.pixelized_image:
            parts.append("pixelized")
        
        if self.dithered_image:
            parts.append("dithered")
            parts.append(self.dither_mode.get())
        
        # Add color count
        try:
            colors = int(self.colors_entry.get())
            parts.append(f"{colors}colors")
        except:
            pass
        
        # Add gamma if enabled
        if self.gamma_var.get():
            parts.append("gamma")
        
        return "_".join(parts) + ".mp4"


def main():
    """Launch the GUI application."""
    app = DitheringApp()  # REMOVE 4 SPACES
    app.mainloop()  # REMOVE 4 SPACES


if __name__ == "__main__":
    main()
