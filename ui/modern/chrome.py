"""Window chrome and navigation promoted from the prototype."""
import tkinter as tk
import customtkinter as ctk
from . import theme as T, winchrome
from .widgets import Icon, Label

def geometry_size(window):
    """Tk's own idea of the size. ``winfo_width`` can report a mid-layout
    value, which would make a dragged window creep by a pixel or two."""
    try:
        size = window.geometry().split("+")[0]
        w, h = size.split("x")
        return int(w), int(h)
    except (ValueError, tk.TclError):
        return window.winfo_width(), window.winfo_height()

def drag_targets(widget):
    """The widget plus whatever CustomTkinter layered on top of it."""
    yield widget
    for attr in ("_canvas", "_label", "_text_label"):
        inner = getattr(widget, attr, None)
        if inner is not None and inner is not widget:
            yield inner

class RailButton(ctk.CTkFrame):
    """Icon over label, with an accent bar marking the active panel."""

    def __init__(self, master, icon, text, command):
        super().__init__(master, fg_color="transparent",
                         corner_radius=T.R_CTRL, width=T.RAIL_W - 12,
                         height=60)
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self._command = command
        self._active = False

        self.icon = Icon(self, icon, 17, T.c("dim"), height=22)
        self.icon.grid(row=0, column=0, pady=(10, 0))
        self.label = Label(self, text, "tiny", T.c("faint"), height=16)
        self.label.grid(row=1, column=0, pady=(2, 10))

        self.bar = ctk.CTkFrame(master, width=2, height=26, corner_radius=0,
                                fg_color=T.accent())

        for w in (self, self.icon, self.label):
            w.bind("<Button-1>", lambda e: self._command())
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _enter(self, _e):
        if not self._active:
            self.configure(fg_color=T.c("card_hi"))

    def _leave(self, _e):
        if not self._active:
            self.configure(fg_color="transparent")

    def set_active(self, on: bool):
        self._active = on
        # Active reads as ink on a raised surface; the accent is just the
        # marker at the edge, so five tinted slabs never compete at once.
        self.configure(fg_color=T.c("card_hi") if on else "transparent")
        self.icon.configure(text_color=T.c("text") if on else T.c("dim"))
        self.label.configure(text_color=T.c("text") if on else T.c("faint"))
        if on:
            self.update_idletasks()
            self.bar.place(x=0, y=self.winfo_y() + 17)
        else:
            self.bar.place_forget()

class Logo(tk.Canvas):
    """The 4x4 Bayer matrix itself, thresholded across a ramp."""

    MATRIX = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))

    def __init__(self, master, size=22):
        super().__init__(master, width=size, height=size, highlightthickness=0,
                         bd=0, bg=T.res(T.c("surface")))
        self._size = size
        self.draw()

    def draw(self):
        self.delete("all")
        self.configure(bg=T.res(T.c("surface")))
        s = self._size
        ink = T.res(T.c("text"))
        step = s / 4.0
        for j in range(4):
            for i in range(4):
                # Left edge is dark, right edge light: the mark is a ramp
                # dithered by the same matrix the app ships with.
                if (1.0 - (i + 0.5) / 4) * 16 > self.MATRIX[j][i]:
                    fill = T.res(T.accent()) if (i + j) == 3 else ink
                    self.create_rectangle(i * step, j * step,
                                          (i + 1) * step, (j + 1) * step,
                                          fill=fill, outline="")

class WindowChrome:
    def _bind_drag(self, widget):
        # A CustomTkinter frame paints on an inner canvas that covers it, and
        # a label wraps a real tk label. Events land on those, not on the
        # widget itself, and tkinter does not bubble to the parent.
        for target in drag_targets(widget):
            target.bind("<Button-1>", self._drag_press, add="+")
            target.bind("<B1-Motion>", self._drag_motion, add="+")
            target.bind("<ButtonRelease-1>", self._drag_release, add="+")
            target.bind("<Double-Button-1>", self._drag_double, add="+")

    def _drag_press(self, event):
        # Arm, do not drag yet: starting the native move loop on press would
        # swallow the double-click that maximizes.
        self._drag_from = (event.x_root, event.y_root)

    def _drag_motion(self, event):
        if not self._custom_chrome:
            return
        if self._manual_drag is not None:
            start_x, start_y, win_x, win_y, win_w, win_h = self._manual_drag
            new_x = win_x + event.x_root - start_x
            new_y = win_y + event.y_root - start_y
            if not winchrome.move_window(self, new_x, new_y):
                # Size is restated every time: a position-only geometry string
                # makes Tk fall back to its requested size mid-drag.
                self.geometry("%dx%d+%d+%d" % (win_w, win_h, new_x, new_y))
            return
        if self._drag_from is None:
            return
        dx = event.x_root - self._drag_from[0]
        dy = event.y_root - self._drag_from[1]
        if abs(dx) < 4 and abs(dy) < 4:
            return
        self._drag_from = None
        origin_x, origin_y = self.winfo_x(), self.winfo_y()
        if self._maximized or self._snapped:
            # Pull the window out from under the cursor at the same relative
            # spot along the bar, the way a native drag-off-maximized does.
            ratio = (event.x_root - origin_x) / max(1, self.winfo_width())
            self._restore_window()
            self.update_idletasks()
            origin_x = int(event.x_root - self.winfo_width() * ratio)
            origin_y = event.y_root - 18
            width, height = geometry_size(self)
            winchrome.move_window(self, origin_x, origin_y)
            self.update_idletasks()
        rect = winchrome.window_rect(self)
        if rect:
            origin_x, origin_y = rect[0], rect[1]
        self._manual_drag = ((event.x_root, event.y_root, origin_x, origin_y)
                             + geometry_size(self))

    def _drag_release(self, event):
        was_dragging = self._manual_drag is not None
        self._drag_from = None
        self._manual_drag = None
        if was_dragging:
            self._snap_at_edge(event)

    def _snap_at_edge(self, event):
        """Our own Aero Snap: the OS version needs its move loop, which the
        interpreter cannot survive being called from."""
        area = winchrome.work_area(self)
        if not area:
            return
        x, y, w, h = area
        margin = 12
        if event.y_root <= y + margin:
            self._maximize_window()
        elif event.x_root <= x + margin:
            self._snap_to(x, y, w // 2, h)
        elif event.x_root >= x + w - margin:
            self._snap_to(x + w // 2, y, w - w // 2, h)

    def _snap_to(self, x, y, w, h):
        # A half-screen narrower than the sidebar and canvas is not usable.
        min_w, min_h = tk.Wm.minsize(self)
        if w < min_w or h < min_h:
            return
        if not (self._maximized or self._snapped):
            self._normal_geometry = self.geometry()
            self._normal_rect = winchrome.window_rect(self)
        self._snapped = True
        self._maximized = False
        winchrome.set_window_rect(self, (x, y, x + w, y + h))
        self.win_buttons["max"].configure(text=T.ICON["win_restore"])
        winchrome.round_corners(self, False)

    def _drag_double(self, _event):
        self._drag_from = None
        self._manual_drag = None
        if self._custom_chrome:
            self.toggle_maximize()

    def set_custom_chrome(self, on: bool):
        """Swap between our title bar and the native one, live."""
        if on == self._custom_chrome:
            return self._custom_chrome
        self._drag_from = self._manual_drag = None
        if on:
            ok = winchrome.hide_caption(self)
            winchrome.round_corners(self, True)
        else:
            if self._maximized or self._snapped:
                self._restore_window()
            ok = False
            winchrome.restore_caption(self)
        self._custom_chrome = ok
        if ok:
            self.win_controls.grid(row=0, column=3, sticky="nse")
        else:
            self.win_controls.grid_remove()
        self.set_status("Title bar: " + ("custom" if ok else "native"))
        return ok

    def act_minimize(self):
        self.iconify()

    def toggle_maximize(self):
        if self._maximized or self._snapped:
            self._restore_window()
        else:
            self._maximize_window()

    def _maximize_window(self):
        area = winchrome.work_area(self)
        if not area:
            return
        if self._maximized:
            return
        if not self._snapped and not self._chrome_transition:
            self._normal_geometry = self.geometry()
            self._normal_rect = winchrome.window_rect(self)
        x, y, w, h = area
        # Driven by hand: a caption-less window maximized by Windows would
        # take the whole monitor and hide the taskbar.
        self._maximized = True
        self._snapped = False
        winchrome.set_window_rect(self, (x, y, x + w, y + h))
        self.win_buttons["max"].configure(text=T.ICON["win_restore"])
        winchrome.round_corners(self, False)

    def _restore_window(self):
        self._chrome_transition = True
        winchrome.round_corners(self, True)
        if self._normal_rect:
            winchrome.set_window_rect(self, self._normal_rect)
        elif self._normal_geometry:
            self.geometry(self._normal_geometry)
        self._maximized = False
        self._snapped = False
        self.win_buttons["max"].configure(text=T.ICON["win_max"])
        self._chrome_transition = False

    def _catch_system_maximize(self, _event=None):
        """Win+Up and drag-to-top ask Windows to zoom; redo it ourselves."""
        if (_event is not None and _event.widget is not self):
            return
        if not self._custom_chrome or self._chrome_transition:
            return
        state = self.state()
        if state == "zoomed":
            if self._system_maximize_after is None:
                self._system_maximize_after = self.after_idle(self._finish_system_maximize)
        elif state == "normal" and not (self._maximized or self._snapped):
            self._normal_rect = winchrome.window_rect(self)
            self._normal_geometry = self.geometry()

    def _finish_system_maximize(self):
        self._system_maximize_after = None
        if not self._custom_chrome or self.state() != "zoomed":
            return
        self._chrome_transition = True
        try:
            self.state("normal")
            self._maximized = False
            self._maximize_window()
        finally:
            self._chrome_transition = False
