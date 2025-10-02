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
import json
import random
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
import numpy as np

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
from config_manager import ConfigManager


class DitheringApp(ctk.CTk):
    """
    Main GUI application with threading to prevent freezing.
    """
    
    def __init__(self):
        super().__init__()
        
        # Load configuration
        self.config = ConfigManager()
        
        # Apply theme settings
        ctk.set_appearance_mode(self.config.get("theme", "appearance_mode", default="system"))
        ctk.set_default_color_theme(self.config.get("theme", "color_theme", default="blue"))
        
        self.title("Image Dithering Tool")
        self.geometry(self.config.get_window_geometry())
        
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
        self.last_gamma = False  # Track gamma setting from last dithering operation
        self.last_pixelization_method = "regular"
        self.display_state = "current"  # Track what we're showing: "current", "pixelized", "dithered"
        self.original_size = None  # Store original image/video size (width, height)
        
        # Pixelization cache tracking
        self.pixelization_cache = {
            "method": None,      # "regular" or "neural"
            "max_size": None,    # The max_size parameter used
            "source_hash": None  # Hash of source image to detect changes
        }
        
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
        
        # Set up close handler to save config
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _compute_image_hash(self, image: Image.Image) -> str:
        """
        Compute a simple hash of the image for cache invalidation.
        Uses image size and a sample of pixels for speed.
        """
        import hashlib
        # Combine size with a sample of pixel data
        data = f"{image.size[0]}x{image.size[1]}"
        # Sample some pixels for uniqueness (faster than hashing entire image)
        arr = np.array(image)
        sample = arr[::50, ::50].tobytes()  # Sample every 50th pixel
        return hashlib.md5((data + str(sample[:1000])).encode()).hexdigest()
    
    def _is_pixelization_cached(self, method: str, max_size: int) -> bool:
        """
        Check if we can reuse the cached pixelized image.
        Returns True if cache is valid, False if re-pixelization needed.
        """
        if self.pixelized_image is None:
            return False
        
        if self.current_image is None:
            return False
        
        # Check if parameters match
        if self.pixelization_cache["method"] != method:
            return False
        
        if self.pixelization_cache["max_size"] != max_size:
            return False
        
        # Check if source image has changed
        current_hash = self._compute_image_hash(self.current_image)
        if self.pixelization_cache["source_hash"] != current_hash:
            return False
        
        return True
    
    def _update_pixelization_cache(self, method: str, max_size: int):
        """Update the pixelization cache after successful pixelization."""
        if self.current_image:
            self.pixelization_cache = {
                "method": method,
                "max_size": max_size,
                "source_hash": self._compute_image_hash(self.current_image)
            }
    
    def _invalidate_pixelization_cache(self):
        """Clear the pixelization cache when source image changes."""
        self.pixelization_cache = {
            "method": None,
            "max_size": None,
            "source_hash": None
        }
        self.pixelized_image = None
    
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
        
        # Random frame button (hidden by default)
        self.random_frame_button = ctk.CTkButton(
            self.sidebar,
            text="Load Random Frame",
            command=self.load_random_frame
        )
        self.random_frame_button.grid(row=row, column=0, pady=5, padx=10, sticky='ew')
        self.random_frame_button.grid_remove()  # Hidden by default
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
        self.max_size_entry.insert(0, str(self.config.get("defaults", "max_size", default=640)))
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
        self.final_resize_var = tk.BooleanVar(
            value=self.config.get("defaults", "final_resize_enabled", default=False)
        )
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
        self.final_size_entry.insert(0, str(self.config.get("defaults", "final_size", default=1920)))
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
        
        self.dither_mode = ctk.StringVar(
            value=self.config.get("defaults", "dither_mode", default="bayer4x4")
        )
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
        self.colors_entry.insert(0, str(self.config.get("defaults", "num_colors", default=16)))
        self.colors_entry.grid(row=row, column=0, pady=2, padx=10, sticky='w')
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
        initial_dir = self.config.get_last_path("image")
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Remember this directory
        self.config.update_last_path("image", filepath)
        self.config.add_recent_file(filepath)
        
        try:
            self.current_image = Image.open(filepath).convert('RGB')
            self.dithered_image = None
            self.is_video = False
            self.video_path = None
            self.display_state = "current"
            
            # Invalidate pixelization cache (new source image)
            self._invalidate_pixelization_cache()
            
            # Store original size and update final size entry
            self.original_size = self.current_image.size  # (width, height)
            larger_dim = max(self.original_size)
            self.final_size_entry.delete(0, tk.END)
            self.final_size_entry.insert(0, str(larger_dim))
            
            # Hide video buttons
            self.apply_video_button.grid_remove()
            self.random_frame_button.grid_remove()
            
            self.image_viewer.set_image(self.current_image)
            self.fit_to_window()
            self.status_bar.set_status(f"Loaded image: {Path(filepath).name} - {self.original_size[0]}x{self.original_size[1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def load_video(self):
        """Load a video file (first frame for preview)."""
        initial_dir = self.config.get_last_path("video")
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Remember this directory
        self.config.update_last_path("video", filepath)
        self.config.add_recent_file(filepath)
        
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
            
            self.dithered_image = None
            self.is_video = True
            self.video_path = filepath
            self.display_state = "current"
            
            # Invalidate pixelization cache (new source image)
            self._invalidate_pixelization_cache()
            
            # Store original size and update final size entry
            self.original_size = self.current_image.size  # (width, height)
            larger_dim = max(self.original_size)
            self.final_size_entry.delete(0, tk.END)
            self.final_size_entry.insert(0, str(larger_dim))
            
            # Show video buttons
            self.apply_video_button.grid()
            self.random_frame_button.grid()
            
            self.image_viewer.set_image(self.current_image)
            self.fit_to_window()
            self.status_bar.set_status(f"Loaded video: {Path(filepath).name} - {self.original_size[0]}x{self.original_size[1]} (showing first frame)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{e}")
    
    def load_random_frame(self):
        """Load a random frame from the video for preview."""
        if not self.is_video or not self.video_path:
            return
        
        try:
            # Get total frame count
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
            
            # Pick random frame (avoiding first frame)
            idx = random.randint(1, total_frames - 1)
            
            # Extract that specific frame
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_frame = os.path.join(tmpdir, "tmp_preview_frame.png")
                ext_cmd = [
                    "ffmpeg", "-y", "-i", self.video_path,
                    "-vf", f"select='eq(n,{idx})'",
                    "-vframes", "1",
                    tmp_frame
                ]
                subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Load the frame
                self.current_image = Image.open(tmp_frame).convert('RGB')
            
            # Reset processing state
            self.dithered_image = None
            self.display_state = "current"
            
            # Invalidate pixelization cache (new frame loaded)
            self._invalidate_pixelization_cache()
            
            # Update display
            self.image_viewer.set_image(self.current_image)
            self.fit_to_window()
            self.status_bar.set_status(f"Loaded random frame #{idx} from video")
            
        except Exception as e:
            messagebox.showerror("Video Error", f"Failed to load random frame:\n{e}")
    
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
        
        # Check if we can reuse cached result
        if self._is_pixelization_cached("regular", max_size):
            self.status_bar.set_status("✓ Using cached pixelization (no re-processing needed)")
            self.last_pixelization_method = "regular"
            self.dithered_image = None
            self.display_state = "pixelized"
            self.image_viewer.set_image(self.pixelized_image)
            self.fit_to_window()
            return
        
        self.status_bar.set_status("Pixelizing...")
        
        # Use threading to prevent GUI freeze
        def process():
            self.pixelized_image = pixelize_regular(self.current_image, max_size)
            self.last_pixelization_method = "regular"
            self.dithered_image = None
            self.display_state = "pixelized"
            
            # Update cache
            self._update_pixelization_cache("regular", max_size)
            
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
        
        # Check if we can reuse cached result
        if self._is_pixelization_cached("neural", max_size):
            self.status_bar.set_status("✓ Using cached neural pixelization (no re-processing needed)")
            self.last_pixelization_method = "neural"
            self.dithered_image = None
            self.display_state = "pixelized"
            self.image_viewer.set_image(self.pixelized_image)
            self.fit_to_window()
            return
        
        self.status_bar.set_status("Neural pixelizing (this may take a moment)...")
        
        def process():
            if self.neural_pix is None:
                self.neural_pix = NeuralPixelizer()
            
            self.pixelized_image = self.neural_pix.pixelize(self.current_image, max_size)
            self.last_pixelization_method = "neural"
            self.dithered_image = None
            self.display_state = "pixelized"
            
            # Update cache
            self._update_pixelization_cache("neural", max_size)
            
            self.after(0, lambda: self.image_viewer.set_image(self.pixelized_image))
            self.after(0, lambda: self.fit_to_window())
            self.after(0, lambda: self.status_bar.set_status("Neural pixelization complete"))
        
        threading.Thread(target=process, daemon=True).start()
    
    def _on_apply_dithering(self):
        """Handle dithering - shows palette selection dialog and applies to current frame."""
        if not self.pixelized_image and not self.current_image:
            messagebox.showwarning("No Image", "Please load an image or video first.")
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
        
        # Use pixelized image if available, otherwise use current image
        source_image = self.pixelized_image if self.pixelized_image else self.current_image
        
        # Show palette selection dialog
        from gui_components import PalettePreview, CustomPaletteCreator, PaletteImagePreviewDialog
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("Select Palette - Live Preview in Main Window")
        
        # Get size and position from config
        dialog_w = self.config.get("ui", "palette_dialog_width", default=400)
        dialog_h = self.config.get("ui", "palette_dialog_height", default=600)
        dialog_x = self.config.get("ui", "palette_dialog_x")
        dialog_y = self.config.get("ui", "palette_dialog_y")
        
        # Set geometry with position if available
        if dialog_x is not None and dialog_y is not None:
            dialog.geometry(f"{dialog_w}x{dialog_h}+{dialog_x}+{dialog_y}")
        else:
            dialog.geometry(f"{dialog_w}x{dialog_h}")
        
        dialog.transient(self)
        dialog.grab_set()
        
        selected_palette = [None]  # Use list to allow modification in nested function
        
        # Store original display state to restore on cancel
        original_displayed_image = None
        if self.display_state == "dithered" and self.dithered_image:
            original_displayed_image = self.dithered_image
        elif self.display_state == "pixelized" and self.pixelized_image:
            original_displayed_image = self.pixelized_image
        else:
            original_displayed_image = self.current_image
        
        # Cache for preview results
        preview_cache = {}
        is_generating = [False]  # Use list for mutable flag
        
        ctk.CTkLabel(dialog, text="Choose Palette:", 
                    font=("Arial", 14, "bold")).pack(pady=2)
        
        ctk.CTkLabel(dialog, text="Preview shows in main window →", 
                    font=("Arial", 11), text_color="gray").pack(pady=(0, 2))
        
        # Gamma correction checkbox in dialog
        gamma_var = tk.BooleanVar(value=self.config.get("defaults", "use_gamma", default=False))
        
        gamma_frame = ctk.CTkFrame(dialog)
        gamma_frame.pack(fill="x", padx=10, pady=5)
        
        gamma_check = ctk.CTkCheckBox(
            gamma_frame,
            text="Gamma Correction",
            variable=gamma_var,
            command=lambda: on_gamma_changed()
        )
        gamma_check.pack(side="left", padx=5)
        
        ctk.CTkLabel(
            gamma_frame,
            text="(affects color accuracy)",
            font=("Arial", 10),
            text_color="gray"
        ).pack(side="left", padx=5)
        
        # Track currently displayed palette for gamma refresh
        current_palette_name = [None]
        current_palette_data = [None]
        
        def save_dialog_size():
            """Save the current dialog size and position to config."""
            try:
                # Get current geometry
                geometry = dialog.geometry()
                # Parse WxH+X+Y format
                parts = geometry.split('+')
                size_part = parts[0]
                w, h = size_part.split('x')
                
                # Save size
                self.config.set("ui", "palette_dialog_width", value=int(w))
                self.config.set("ui", "palette_dialog_height", value=int(h))
                
                # Save position if available
                if len(parts) >= 3:
                    self.config.set("ui", "palette_dialog_x", value=int(parts[1]))
                    self.config.set("ui", "palette_dialog_y", value=int(parts[2]))
            except Exception as e:
                print(f"Error saving dialog size/position: {e}")
        
        def on_gamma_changed():
            """Regenerate preview when gamma checkbox is toggled."""
            # Clear cache since gamma affects results
            preview_cache.clear()
            
            # Regenerate current preview if one is selected
            if current_palette_name[0] and current_palette_data[0]:
                threading.Thread(
                    target=lambda: generate_preview(current_palette_name[0], current_palette_data[0]),
                    daemon=True
                ).start()
        
        def generate_preview(palette_name, palette):
            """Generate preview and display in MAIN viewer window."""
            if is_generating[0]:
                return  # Skip if already generating
            
            # Create cache key that includes gamma state
            cache_key = f"{palette_name}_gamma{gamma_var.get()}"
            
            # Check cache first
            if cache_key in preview_cache:
                self.after(0, lambda: display_preview(preview_cache[cache_key]))
                self.after(0, lambda: self.status_bar.set_status(f"Preview: {palette_name} (cached)"))
                return
            
            is_generating[0] = True
            self.after(0, lambda: self.status_bar.set_status(f"Generating preview: {palette_name}..."))
            
            try:
                # Store current palette info for gamma refresh
                current_palette_name[0] = palette_name
                current_palette_data[0] = palette
                
                # Get current dither mode
                try:
                    dither_mode = DitherMode(self.dither_mode.get())
                except:
                    dither_mode = DitherMode.BAYER4x4
                
                # Create ditherer with gamma from dialog
                ditherer = ImageDitherer(
                    num_colors=num_colors,
                    dither_mode=dither_mode,
                    palette=palette,
                    use_gamma=gamma_var.get()
                )
                
                # Apply dithering to source image (full resolution!)
                preview_result = ditherer.apply_dithering(source_image)
                
                # Apply final resize if enabled
                preview_result = self._apply_final_resize(preview_result)
                
                # Cache the result with gamma in key
                preview_cache[cache_key] = preview_result
                
                # Display in MAIN viewer
                self.after(0, lambda: display_preview(preview_result))
                gamma_text = " (gamma)" if gamma_var.get() else ""
                self.after(0, lambda: self.status_bar.set_status(f"Preview: {palette_name}{gamma_text}"))
                
            except Exception as e:
                print(f"Preview generation error: {e}")
                self.after(0, lambda: self.status_bar.set_status(f"Preview error: {str(e)[:50]}"))
            finally:
                is_generating[0] = False
        
        def display_preview(preview_img):
            """Display preview in the MAIN image viewer."""
            self.image_viewer.set_image(preview_img)
            self.fit_to_window()
        
        def on_palette_selected(palette_name):
            """Callback when palette is selected - trigger preview generation."""
            # Find the palette
            for name, palette in palette_options:
                if name == palette_name:
                    # Generate preview in background thread
                    threading.Thread(
                        target=lambda: generate_preview(name, palette),
                        daemon=True
                    ).start()
                    break
        
        # Generate palette options
        def generate_palette_options():
            palette_options = []
            
            # Median Cut
            mc_palette = ColorReducer.reduce_colors(source_image, num_colors)
            palette_options.append(("Median Cut", mc_palette))
            
            # K-means
            km_palette = ColorReducer.generate_kmeans_palette(source_image, num_colors, random_state=42)
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
            
            return palette_options
        
        palette_options = generate_palette_options()
        
        # Radio buttons for selection
        selected_var = tk.StringVar(value="Median Cut")
        
        scroll_frame = ctk.CTkScrollableFrame(dialog, height=350)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        def update_palette_list():
            # Clear existing widgets
            for widget in scroll_frame.winfo_children():
                widget.destroy()
            
            nonlocal palette_options
            palette_options = generate_palette_options()
            
            # Clear preview cache when palette list changes
            preview_cache.clear()
            
            for name, palette in palette_options:
                frame = ctk.CTkFrame(scroll_frame)
                frame.pack(fill="x", padx=5, pady=5)
                
                radio = ctk.CTkRadioButton(
                    frame, 
                    text=name, 
                    variable=selected_var, 
                    value=name,
                    command=lambda n=name: on_palette_selected(n)
                )
                radio.pack(side="left", padx=5)
                
                # Show palette preview bar
                preview = PalettePreview(frame, palette, width=220, height=20)
                preview.pack(side="right", padx=5)
        
        update_palette_list()
        
        # Generate preview for default selection (Median Cut)
        for name, palette in palette_options:
            if name == "Median Cut":
                threading.Thread(
                    target=lambda: generate_preview(name, palette),
                    daemon=True
                ).start()
                break
        
        # Palette management buttons
        custom_buttons_frame = ctk.CTkFrame(dialog)
        custom_buttons_frame.pack(pady=10, fill='x', padx=10)
        
        def create_custom_palette():
            CustomPaletteCreator(dialog, self.palette_mgr, update_palette_list)
        
        def import_from_lospec():
            self._import_from_lospec(dialog, update_palette_list)
        
        def create_palette_from_image():
            self._create_palette_from_image(dialog, update_palette_list, selected_var)
        
        ctk.CTkButton(
            custom_buttons_frame,
            text="Custom",
            command=create_custom_palette,
            width=80
        ).pack(side="left", padx=2, fill='x', expand=True)
        
        ctk.CTkButton(
            custom_buttons_frame,
            text="Lospec",
            command=import_from_lospec,
            width=80
        ).pack(side="left", padx=2, fill='x', expand=True)
        
        ctk.CTkButton(
            custom_buttons_frame,
            text="From Image",
            command=create_palette_from_image,
            width=80
        ).pack(side="left", padx=2, fill='x', expand=True)
        
        def on_ok():
            # Save dialog size before closing
            save_dialog_size()
            
            # Find selected palette
            sel_name = selected_var.get()
            for name, palette in palette_options:
                if name == sel_name:
                    selected_palette[0] = palette
                    break
            
            # The preview is already showing in the main window
            # If it's cached, use that, otherwise it will be generated below
            cache_key = f"{sel_name}_gamma{gamma_var.get()}"
            if cache_key in preview_cache:
                self.dithered_image = preview_cache[cache_key]
                self.display_state = "dithered"
            
            dialog.destroy()
        
        def on_cancel():
            # Save dialog size before closing
            save_dialog_size()
            
            # Restore original display
            if original_displayed_image:
                self.image_viewer.set_image(original_displayed_image)
                self.fit_to_window()
            self.status_bar.set_status("Palette selection cancelled")
            dialog.destroy()
        
        # Handle window close (X button) same as Cancel
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        
        btn_frame = ctk.CTkFrame(dialog)
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="Cancel", command=on_cancel, width=120).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Apply Selected", command=on_ok, width=120).pack(side="right", padx=5)
        
        # Wait for dialog to close
        self.wait_window(dialog)
        
        if selected_palette[0] is None:
            return  # User cancelled
        
        # Store the palette and gamma setting for video processing
        self.last_palette = selected_palette[0]
        self.last_gamma = gamma_var.get()
        
        # Save gamma to config as default for next time
        self.config.set("defaults", "use_gamma", value=gamma_var.get())
        
        # If we already have the result cached from preview, we're done!
        # The on_ok() already set self.dithered_image from cache
        # Just need to handle the case where it wasn't cached (shouldn't happen normally)
        if self.dithered_image is None or self.display_state != "dithered":
            # Fallback: generate it now (shouldn't be reached if preview worked)
            try:
                dither_mode = DitherMode(self.dither_mode.get())
            except:
                dither_mode = DitherMode.BAYER4x4
            
            self.status_bar.set_status("Applying final dithering...")
            
            def process():
                ditherer = ImageDitherer(
                    num_colors=num_colors,
                    dither_mode=dither_mode,
                    palette=selected_palette[0],
                    use_gamma=gamma_var.get()
                )
                
                # Use pixelized image if available, otherwise use current image
                source_for_dithering = self.pixelized_image if self.pixelized_image else self.current_image
                self.dithered_image = ditherer.apply_dithering(source_for_dithering)
                
                # Apply final resize if enabled
                self.dithered_image = self._apply_final_resize(self.dithered_image)
                self.display_state = "dithered"
                
                self.after(0, lambda: self.image_viewer.set_image(self.dithered_image))
                self.after(0, lambda: self.fit_to_window())
                self.after(0, lambda: self.status_bar.set_status("Dithering complete"))
            
            threading.Thread(target=process, daemon=True).start()
        else:
            # We have the cached result, just update status
            self.status_bar.set_status("Dithering applied from preview")
    
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
        
        initial_dir = self.config.get_last_path("save")
        output_path = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            defaultextension=".mp4",
            initialfile=default_video_name,
            filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Remember this directory
        self.config.update_last_path("save", output_path)
        
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
                use_gamma=self.last_gamma
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
        
        initial_dir = self.config.get_last_path("save")
        filepath = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            # Remember this directory
            self.config.update_last_path("save", filepath)
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
            elif self.dithered_image:
                # Skip directly to dithered if no pixelization step
                self.image_viewer.set_image(self.dithered_image)
                self.display_state = "dithered"
                self.fit_to_window()
            else:
                messagebox.showinfo("No Processed Image", "Please pixelize or apply dithering first.")
        elif self.display_state == "pixelized":
            if self.dithered_image:
                self.image_viewer.set_image(self.dithered_image)
                self.display_state = "dithered"
                self.fit_to_window()
            else:
                messagebox.showinfo("No Dithered Image", "Please apply dithering first.")
        elif self.display_state == "dithered":
            # Go back to pixelized if it exists, otherwise go to current
            if self.pixelized_image:
                self.image_viewer.set_image(self.pixelized_image)
                self.display_state = "pixelized"
            else:
                self.image_viewer.set_image(self.current_image)
                self.display_state = "current"
            self.fit_to_window()
    
    def _import_from_lospec(self, parent_dialog, refresh_callback):
        """Import palette from lospec.com URL."""
        from tkinter import simpledialog
        import urllib.request
        import json
        
        url = simpledialog.askstring("Import Palette", "Paste lospec.com Palette URL:", parent=parent_dialog)
        if not url:
            return
        
        try:
            parts = url.rstrip('/').split('/')
            if len(parts) < 2:
                raise ValueError("URL does not contain enough parts to extract palette name.")
            palette_slug = parts[-1]
            json_url = f'https://lospec.com/palette-list/{palette_slug}.json'
        except Exception as e:
            messagebox.showerror("Invalid URL", f"Failed to parse palette name:\n{e}", parent=parent_dialog)
            return
        
        try:
            with urllib.request.urlopen(json_url) as resp:
                data = resp.read()
                pjson = json.loads(data)
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download or parse palette JSON:\n{e}", parent=parent_dialog)
            return
        
        try:
            name = pjson['name']
            colors = pjson['colors']
            hex_colors = [f"#{c}" for c in colors]
        except KeyError as e:
            messagebox.showerror("JSON Error", f"Missing key in palette JSON: {e}", parent=parent_dialog)
            return
        except Exception as e:
            messagebox.showerror("Parse Error", f"Failed to parse palette JSON:\n{e}", parent=parent_dialog)
            return
        
        # Check for duplicates
        existing_names = self.palette_mgr.list_palette_names()
        if name in existing_names:
            messagebox.showerror("Duplicate Palette", f"A palette named '{name}' already exists.", parent=parent_dialog)
            return
        
        self.palette_mgr.add_palette(name, hex_colors)
        refresh_callback()
        messagebox.showinfo("Success", f"Palette '{name}' imported successfully.", parent=parent_dialog)
    
    def _create_palette_from_image(self, parent_dialog, refresh_callback, selected_var):
        """Create palette from image file using k-means clustering."""
        from gui_components import PaletteImagePreviewDialog
        from sklearn.cluster import KMeans
        
        fp = filedialog.askopenfilename(
            parent=parent_dialog,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                      ("All files", "*.*")]
        )
        if not fp:
            return
        
        try:
            new_img = Image.open(fp).convert('RGB')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}", parent=parent_dialog)
            return
        
        try:
            desired = int(self.colors_entry.get())
            if desired <= 0:
                raise ValueError
        except:
            desired = 16
        
        arr_full = np.array(new_img)
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
        
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(small)
        centers = kmeans.cluster_centers_.astype(int)
        kpal = [tuple(v) for v in centers]
        
        from_img_preview = PaletteImagePreviewDialog(parent_dialog, kpal, fp, used_clusters=n)
        parent_dialog.wait_window(from_img_preview)
        
        if from_img_preview.choose_another:
            # Recursively call to pick another image
            self._create_palette_from_image(parent_dialog, refresh_callback, selected_var)
            return
        elif from_img_preview.use_result:
            # Generate unique name with random hex + color count
            import secrets
            random_hex = secrets.token_hex(3)  # 6-character hex string
            pname = f"img_{random_hex}_{n}colors"
            
            hex_colors = [f'#{col[0]:02x}{col[1]:02x}{col[2]:02x}' for col in kpal]
            self.palette_mgr.add_palette(pname, hex_colors)
            refresh_callback()
            selected_var.set(pname)
    
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
    
    def _on_closing(self):
        """Handle window close event - save configuration."""
        try:
            # Save window geometry
            geometry = self.geometry()
            # Check if maximized (state returns 'zoomed' on Windows when maximized)
            is_maximized = self.state() == 'zoomed'
            self.config.save_window_geometry(geometry, is_maximized)
            
            # Save current settings as defaults
            try:
                self.config.set("defaults", "max_size", value=int(self.max_size_entry.get()))
            except:
                pass
            
            try:
                self.config.set("defaults", "num_colors", value=int(self.colors_entry.get()))
            except:
                pass
            
            try:
                self.config.set("defaults", "final_size", value=int(self.final_size_entry.get()))
            except:
                pass
            
            self.config.set("defaults", "dither_mode", value=self.dither_mode.get())
            # Gamma is saved when user selects it in the palette dialog
            self.config.set("defaults", "final_resize_enabled", value=self.final_resize_var.get())
            
            # Save configuration to file
            self.config.save()
        except Exception as e:
            print(f"Error saving config: {e}")
        
        # Close the application
        self.destroy()


def main():
    """Launch the GUI application."""
    app = DitheringApp()  # REMOVE 4 SPACES
    app.mainloop()  # REMOVE 4 SPACES


if __name__ == "__main__":
    main()
