"""Real image/video workflows for the modern shell."""
from pathlib import Path
from queue import Empty
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from dithering_lib import DitherMode, PixelizeMethod
from pixelization import NeuralPixelizer, pixelize_regular
from utils import has_actual_transparency
from video_processor import VideoProcessor
from .processing import DitherRequest, GENERATED, read_image, read_video_frame


class Operations:
    def _poll_work(self):
        if self._shutdown_requested:
            self._shutdown_requested = False
            self._on_closing()
        if self._closing:
            return
        self._image_jobs.poll()
        self._preview_jobs.poll()
        self._palette_jobs.poll()
        self._video_jobs.poll()
        try:
            while True:
                fraction, text = self._progress.get_nowait()
                if self._video_busy:
                    self.status_bar.show_progress(fraction)
                    self.set_status(text, 'busy')
        except Empty:
            pass
        self.after(30, self._poll_work)

    def _error(self, error):
        self._refresh_capabilities()
        self.set_status(str(error), 'err')
        messagebox.showerror('Processing failed', str(error), parent=self)

    def _cancel_preview(self):
        self._preview_jobs.invalidate()
        if self._preview_after is not None:
            self.after_cancel(self._preview_after)
            self._preview_after = None

    def _request_preview(self, *_, activate=False):
        self._cancel_preview()
        self._preview_dirty = True
        if self.current_image is None or self._image_busy:
            return
        if activate:
            self._preview_enabled_for_source = True
        if activate or self.live_var.get():
            self._preview_after = self.after(180, lambda: self._on_apply_dithering(activate=True))
            self.set_status('Preparing preview…', 'busy')
        else:
            self.set_status('Settings changed — update preview', 'warn')

    def _request_snapshot(self):
        name = self.palette_var.get()
        colors = self._selected_palette_colors()
        if name not in GENERATED and not colors:
            raise ValueError('Select a nonempty palette.')
        return DitherRequest(self.dither_mode.get(), int(self.colors_entry.get()),
                             self.gamma_var.get(), name,
                             None if colors is None else tuple(colors),
                             dict(self.dither_parameters.get(self.dither_mode.get(), {})))

    def _on_apply_dithering(self, activate=True):
        self._cancel_preview()
        if self.current_image is None or self._image_busy:
            return
        try:
            self._commit_numbers()
            self._cancel_preview()
            request = self._request_snapshot()
        except ValueError as error:
            self._error(error)
            return
        source = self.pixelized_image or self.current_image
        self._preview_enabled_for_source = True
        self._preview_dirty = True
        view_revision = self._view_revision
        self.set_status('Generating preview…', 'busy')

        def done(result):
            self.dithered_image, self.last_palette = result
            self.last_gamma = request.gamma
            self._rendered_request = request
            self._preview_dirty = False
            if request.palette_name in GENERATED:
                self._generated_palettes[request.palette_name] = self.last_palette
                self._filter_palettes()
            if view_revision == self._view_revision and (activate or self.display_state == 'dithered'):
                self.show_view('Dithered')
            self._update_resize_preview()
            self.set_status(f'Applied: {request.palette_name} · {request.mode}')
        self._preview_jobs.submit(lambda: request.render(source), done, self._error)

    def _begin_image_job(self, compute, on_result, text):
        self._cancel_preview()
        self._preview_dirty = True
        self._image_busy = True
        self._refresh_capabilities()
        self.set_status(text, 'busy')
        def done(value):
            self._image_busy = False
            on_result(value)
            self._refresh_capabilities()
        def failed(error):
            self._image_busy = False
            self._error(error)
        self._image_jobs.submit(compute, done, failed)

    def load_image(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(parent=self,
                initialdir=self.config.get_last_path('image'),
                filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tiff'), ('All files', '*.*')])
        if path:
            self._begin_image_job(lambda: read_image(path),
                lambda image: self._source_loaded(image, path), 'Loading image…')

    def load_video(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(parent=self,
                initialdir=self.config.get_last_path('video'),
                filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv *.webm'), ('All files', '*.*')])
        if not path:
            return
        def compute():
            info = VideoProcessor().get_video_info(path)
            return read_video_frame(path, info), info
        self._begin_image_job(compute,
            lambda result: self._source_loaded(result[0], path, result[1]), 'Loading video frame…')

    def _source_loaded(self, image, path, video_info=None):
        self._cancel_preview()
        self.current_image = image
        self.is_video = video_info is not None
        self.video_path = str(path) if self.is_video else None
        self._video_info = video_info
        self._last_loaded_file = str(path)
        self.original_size = image.size
        self.source_has_alpha = has_actual_transparency(image)
        self._invalidate_pixelization_cache()
        self._pixelize_spec = None
        self.dithered_image = self.last_palette = None
        self._rendered_request = None
        self._preview_dirty = False
        self._preview_enabled_for_source = False
        self.config.update_last_path('video' if self.is_video else 'image', str(path))
        self.config.add_recent_file(str(path))
        self.file_label.configure(text=Path(path).name)
        self.source_label.configure(text=f'{image.width} × {image.height}' +
            (f"  ·  {video_info['fps']:.3f} fps" if self.is_video else
             ('  ·  transparency' if self.source_has_alpha else '  ·  RGB')))
        self.show_view('Original', fit=True)
        self._update_resize_preview()
        self._refresh_recent()
        self._refresh_palettes()
        self.set_status(f'Loaded {Path(path).name}')

    def load_random_frame(self):
        if not self.is_video or not self._video_info:
            return
        count = self._video_info.get('frame_count')
        if not count:
            self.set_status('Video has no known frame count', 'warn')
            return
        index = random.randrange(count)
        path, info = self.video_path, self._video_info
        self._begin_image_job(lambda: read_video_frame(path, info, index),
            lambda image: self._source_loaded(image, path, info), f'Loading frame {index}…')

    def _pixelize(self, method):
        if self.current_image is None or self._image_busy:
            return
        if method == PixelizeMethod.NEURAL and self.source_has_alpha:
            self.set_status('Neural pixelization cannot preserve transparency', 'warn')
            return
        self._commit_numbers()
        size = int(self.max_size_entry.get())
        source = self.current_image
        def compute():
            return (NeuralPixelizer().pixelize(source, size) if method == PixelizeMethod.NEURAL
                    else pixelize_regular(source, size))
        def done(image):
            self.pixelized_image = image
            self.dithered_image = None
            self.last_palette = self._rendered_request = None
            self._preview_enabled_for_source = False
            self._preview_dirty = False
            self.last_pixelization_method = method
            self._pixelize_spec = (method.value, size)
            self._update_pixelization_cache(method, size)
            self.show_view('Pixelized')
            self._update_resize_preview()
            self.set_status('Pixelization complete')
            self._refresh_palettes()
        self._begin_image_job(compute, done, 'Pixelizing…')

    def _on_pixelize_regular(self):
        self._pixelize(PixelizeMethod.REGULAR)

    def _on_pixelize_neural(self):
        self._pixelize(PixelizeMethod.NEURAL)

    def _on_pixelization_editor(self):
        if self.current_image is None or self.is_video or self._image_busy:
            return
        self._cancel_preview()
        from .editor import ModernPixelizationEditor
        dialog = ModernPixelizationEditor(self, self.current_image, config=self.config,
                                           alpha_cutoff=self._get_alpha_cutoff())
        self.wait_window(dialog)
        if dialog.result_image is None:
            return
        result = dialog.result_image
        self.source_has_alpha = has_actual_transparency(result)
        self.pixelized_image = result.convert('RGBA' if self.source_has_alpha else 'RGB')
        self.last_pixelization_method = PixelizeMethod.REGULAR
        self.pixelization_cache = {'method': 'editor', 'max_size': None,
            'source_hash': self._compute_image_hash(self.current_image)}
        self._pixelize_spec = None
        self.dithered_image = self.last_palette = self._rendered_request = None
        self._preview_enabled_for_source = self._preview_dirty = False
        self.show_view('Pixelized')
        self._update_resize_preview()
        self._refresh_palettes()
        self._refresh_capabilities()
        self.set_status('Pixelization editor applied')

    def _use_original(self):
        self._image_jobs.invalidate()
        self._image_busy = False
        self._cancel_preview()
        self._invalidate_pixelization_cache()
        self._pixelize_spec = None
        self.dithered_image = None
        self._preview_enabled_for_source = False
        self._preview_dirty = False
        self.last_palette = self._rendered_request = None
        self.show_view('Original')
        self._refresh_capabilities()
        self._refresh_palettes()

    def save_image(self):
        self._commit_numbers()
        if self._image_busy or (self.dithered_image is not None and self._preview_dirty):
            self.set_status('Update the preview before saving', 'warn')
            return
        super().save_image()

    def _apply_to_video_workflow(self):
        self._commit_numbers()
        if self._video_busy:
            return
        if not self.video_path or not self.last_palette or self._preview_dirty:
            self.set_status('Generate a current dithered preview before exporting video', 'warn')
            return
        self._commit_numbers()
        path = filedialog.asksaveasfilename(parent=self,
            initialdir=self.config.get_last_path('save'), defaultextension='.mp4',
            initialfile=self._generate_video_filename(), filetypes=[('MP4 video', '*.mp4')])
        if not path:
            return
        self.config.update_last_path('save', path)
        source = self.video_path
        request = self._rendered_request
        palette = list(self.last_palette)
        spec = self._pixelize_spec
        multiplier = int(self.resize_multiplier_entry.get()) if self.final_resize_var.get() else None
        workers = self.workers.get()
        self._cancel_video.clear()
        self._video_busy = True
        self.cancel_video_button.configure(state='normal')
        self.apply_video_button.configure(state='disabled')

        def progress(fraction, message):
            if self._cancel_video.is_set():
                raise RuntimeError('Video processing cancelled')
            self._progress.put((fraction, message))
        def compute():
            processor = VideoProcessor(workers, progress)
            return processor.process_video_streaming(source, path, request.ditherer(palette),
                pixelize_func=spec, final_resize_multiplier=multiplier)
        def done(ok):
            self._video_busy = False
            self.status_bar.hide_progress()
            self.cancel_video_button.configure(state='disabled')
            self._refresh_capabilities()
            self.set_status('Video saved' if ok else ('Video cancelled' if self._cancel_video.is_set()
                                                     else 'Video failed; see console for details'),
                            'idle' if ok else 'warn')
        def failed(error):
            done(False)
            if not self._cancel_video.is_set():
                self._error(error)
        self._video_jobs.submit(compute, done, failed)

    def _refresh_capabilities(self):
        ready = self.current_image is not None and not self._image_busy
        for widget in (self.btn_pixelize, self.btn_dither, self.btn_save):
            widget.configure(state='normal' if ready else 'disabled')
        self.btn_pixelize_neural.configure(state='normal' if ready and not self.source_has_alpha else 'disabled')
        self.btn_pixelize_editor.configure(state='normal' if ready and not self.is_video else 'disabled')
        self.random_frame_button.configure(state='normal' if ready and self.is_video else 'disabled')
        self.apply_video_button.configure(state='normal' if ready and self.is_video and not self._video_busy else 'disabled')
