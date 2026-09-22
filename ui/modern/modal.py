"""Shared custom dialog frame matching the design prototype."""
import tkinter as tk
import customtkinter as ctk
from . import theme as T, winchrome
from .widgets import Icon, Label

class Dialog(winchrome.StableTitlebar, ctk.CTkToplevel):
    """Frameless-feeling dialog: own header row, body area, footer actions."""

    def __init__(self, parent, title: str, icon: str = None,
                 size=(520, 420), resizable=False):
        super().__init__(parent)
        # The in-window header is the real title, so the native bar just
        # carries the app name instead of repeating it.
        self.title("Dither Pie")
        self.geometry("%dx%d" % size)
        self.minsize(360, 260)
        self.resizable(resizable, resizable)
        self._resizable = resizable
        self.configure(fg_color=T.c("bg"))
        self.transient(parent)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent", height=46)
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 0))
        head.grid_columnconfigure(1, weight=1)
        col = 0
        if icon:
            Icon(head, icon, 15, T.c("faint")).grid(row=0, column=0,
                                                       padx=(0, 9))
            col = 1
        Label(head, title, "h1").grid(row=0, column=col, sticky="w")
        self._close_btn = ctk.CTkButton(
            head, text=T.ICON["win_close"], width=32, height=28,
            corner_radius=T.R_CTRL, font=T.icon_font(11),
            fg_color="transparent", text_color=T.c("dim"),
            hover_color=T.c("err"), command=self.close)
        self._close_btn.bind("<Enter>", lambda e: self._close_btn.configure(
            text_color=("#FFFFFF", "#FFFFFF")), add="+")
        self._close_btn.bind("<Leave>", lambda e: self._close_btn.configure(
            text_color=T.c("dim")), add="+")
        self._head = head

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(14, 0))
        self.body.grid_columnconfigure(0, weight=1)
        # Subclasses declare which of their rows absorbs the slack.

        self.footer = ctk.CTkFrame(self, fg_color="transparent")
        self.footer.grid(row=2, column=0, sticky="ew", padx=18, pady=16)
        self.footer.grid_columnconfigure(0, weight=1)

        self.bind("<Escape>", lambda e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._drag_from = None
        self._manual_drag = None
        self._startup_ids = [self.after(60, self._center),
                             self.after(120, self.grab_set),
                             self.after(90, self._apply_chrome)]

    def _apply_chrome(self):
        """Match the main window: drop the native caption, drag by the header."""
        if not getattr(self.master, "_custom_chrome", False):
            winchrome.restore_caption(self)
            self._close_btn.grid_remove()
            return
        if not winchrome.hide_caption(self, resizable=self._resizable):
            return
        winchrome.round_corners(self, True)
        self._close_btn.grid(row=0, column=2, sticky="e")
        from .chrome import drag_targets
        for widget in (self._head,) + tuple(self._head.winfo_children()):
            if widget is self._close_btn:
                continue
            for target in drag_targets(widget):
                target.bind("<Button-1>", self._drag_press, add="+")
                target.bind("<B1-Motion>", self._drag_motion, add="+")
                target.bind("<ButtonRelease-1>", self._drag_release, add="+")

    def _drag_press(self, event):
        self._drag_from = (event.x_root, event.y_root)

    def _drag_motion(self, event):
        if self._manual_drag is not None:
            start_x, start_y, win_x, win_y, win_w, win_h = self._manual_drag
            new_x = win_x + event.x_root - start_x
            new_y = win_y + event.y_root - start_y
            if not winchrome.move_window(self, new_x, new_y):
                self.geometry("%dx%d+%d+%d" % (win_w, win_h, new_x, new_y))
            return
        if self._drag_from is None:
            return
        if (abs(event.x_root - self._drag_from[0]) < 4
                and abs(event.y_root - self._drag_from[1]) < 4):
            return
        self._drag_from = None
        from .chrome import geometry_size
        rect = winchrome.window_rect(self)
        origin = (rect[0], rect[1]) if rect else (self.winfo_x(),
                                                  self.winfo_y())
        self._manual_drag = ((event.x_root, event.y_root) + origin
                             + geometry_size(self))

    def _drag_release(self, _event):
        self._drag_from = None
        self._manual_drag = None

    def _center(self):
        self.update_idletasks()
        p = self.master
        x = p.winfo_rootx() + (p.winfo_width() - self.winfo_width()) // 2
        y = p.winfo_rooty() + (p.winfo_height() - self.winfo_height()) // 3
        self.geometry("+%d+%d" % (max(0, x), max(0, y)))

    def add_actions(self, *buttons):
        for i, b in enumerate(buttons):
            b.grid(row=0, column=i + 1, padx=(8, 0))

    def close(self):
        for token in self._startup_ids:
            self.after_cancel(token)
        self._startup_ids.clear()
        self.grab_release()
        self.destroy()
