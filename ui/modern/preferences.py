"""Preferences with the prototype's theme cards and color samples."""
import customtkinter as ctk
from . import theme as T
from .modal import Dialog
from .widgets import Btn, Card, Divider, Icon, Label, Segmented, SliderField, Toggle

class Preferences(Dialog):
    def __init__(self, app):
        super().__init__(app, "Preferences", "settings", size=(520, 620),
                         resizable=True)
        self.app = app
        self.body.grid_rowconfigure(0, weight=1)
        b = ctk.CTkScrollableFrame(self.body, fg_color="transparent",
                                   scrollbar_button_color=T.c("border"))
        b.grid(row=0, column=0, sticky="nsew")
        b.grid_columnconfigure(0, weight=1)

        look = Card(b, "Appearance", "sun")
        look.grid(row=0, column=0, sticky="ew")
        lb = look.body
        Label(lb, T.caps("Mode"), "eyebrow", T.c("dim")).grid(
            row=0, column=0, sticky="w")
        Segmented(lb, ["Light", "Dark", "System"],
                  value=app.config.get("theme", "appearance_mode").capitalize(),
                  icons=["sun", "moon", "monitor"],
                  command=self._set_appearance).grid(
            row=1, column=0, sticky="ew", pady=(6, 0))

        Label(lb, T.caps("Design system"), "eyebrow", T.c("dim")).grid(
            row=2, column=0, sticky="w", pady=(18, 7))
        systems = ctk.CTkFrame(lb, fg_color="transparent")
        systems.grid(row=3, column=0, sticky="ew")
        systems.grid_columnconfigure(0, weight=1)
        self._system_rows = {}
        self._system_swatches = {}

        def choose_system(name):
            app.set_design_system(name)
            for label, (row, marks) in self._system_rows.items():
                on = label == name
                row.configure(border_color=T.accent() if on else T.c("border"),
                              fg_color=T.c("card_hi") if on else "transparent")
                marks.configure(text=T.ICON["check"] if on else "",
                                text_color=T.accent_text())

        for i, name in enumerate(T.SYSTEMS):
            selected = name == T.system_name()
            row = ctk.CTkFrame(
                systems, corner_radius=T.R_CTRL,
                fg_color=T.c("card_hi") if selected else "transparent",
                border_width=1,
                border_color=T.accent() if selected else T.c("border"))
            row.grid(row=i, column=0, sticky="ew", pady=(0 if i == 0 else 6, 0))
            row.grid_columnconfigure(1, weight=1)

            # Three anchors of each system, shown in the mode being used.
            spec = T.SYSTEMS[name]["dark" if T.is_dark() else "light"]
            keys = ("base", "ink", "accent") if T.is_dark()                 else ("paper", "ink", "accent")
            chips = ctk.CTkFrame(row, fg_color="transparent")
            chips.grid(row=0, column=0, rowspan=2, padx=(10, 11), pady=10)
            self._system_swatches[name] = []
            for j, key in enumerate(keys):
                swatch = ctk.CTkFrame(chips, width=14, height=26,
                                      corner_radius=0, fg_color=spec[key],
                                      border_width=1,
                                      border_color=T.c("border"))
                swatch._fixed_accent = True
                self._system_swatches[name].append(swatch)
                swatch.grid(row=0, column=j)
                swatch.grid_propagate(False)

            Label(row, name, "strong").grid(row=0, column=1, sticky="w",
                                            pady=(9, 0))
            Label(row, T.system_note(name), "tiny", T.c("faint"),
                  wraplength=250, justify="left").grid(
                row=1, column=1, sticky="w", pady=(1, 9))
            mark = Icon(row, "check" if selected else "blank", 12,
                        T.accent_text())
            mark.configure(text=T.ICON["check"] if selected else "")
            mark.grid(row=0, column=2, rowspan=2, padx=(6, 11))

            self._system_rows[name] = (row, mark)
            self._bind_choice(row, lambda e, n=name: choose_system(n))

        Label(lb, "Palettes from designdotmd.directory, which catalogs "
                  "DESIGN.md token sets with their reasoning.",
              "tiny", T.c("faint"), wraplength=380, justify="left").grid(
            row=4, column=0, sticky="w", pady=(10, 0))

        beh = Card(b, "Behaviour", "settings")
        beh.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        chrome = Toggle(beh.body, "Custom title bar", value=app._custom_chrome)
        def set_chrome():
            app.set_custom_chrome(chrome.get())
            self._apply_chrome()
        chrome.switch.configure(command=set_chrome)
        chrome.grid(row=0, column=0, sticky="ew")
        Divider(beh.body).grid(row=1, column=0, sticky="ew", pady=12)
        live = Toggle(beh.body, "Live preview updates", "Update when gamma, size or parameters change.", app.live_var.get())
        live.switch.configure(command=lambda: app.live_var.set(live.get()))
        live.grid(row=2, column=0, sticky="ew")
        Divider(beh.body).grid(row=3, column=0, sticky="ew", pady=12)
        grid = Toggle(beh.body, "Pixel grid at high zoom", value=app.image_viewer.show_pixel_grid)
        def set_grid():
            app.image_viewer.show_pixel_grid = grid.get()
            app.image_viewer.redraw()
        grid.switch.configure(command=set_grid)
        grid.grid(row=4, column=0, sticky="ew")
        perf = Card(b, "Processing", "timer")
        perf.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        def set_workers(value):
            app.workers.var.set(value)
            app.workers._sync_entry()
        SliderField(perf.body, "Worker processes", 1, 32, app.workers.get(), 1,
                    typed_bounds=(1, None), command=set_workers).grid(row=0, column=0, sticky="ew")
        self.add_actions(Btn(self.footer, "Done", variant="primary", command=self.close, width=110))

    def _bind_choice(self, widget, callback):
        widget.bind("<Button-1>", callback, add="+")
        for child in widget.winfo_children():
            self._bind_choice(child, callback)

    def _set_appearance(self, mode):
        self.app.set_appearance(mode)
        self.refresh_samples()

    def refresh_samples(self):
        mode = "dark" if T.is_dark() else "light"
        keys = ("base", "ink", "accent") if T.is_dark() else ("paper", "ink", "accent")
        for name, swatches in self._system_swatches.items():
            for swatch, key in zip(swatches, keys):
                swatch.configure(fg_color=T.SYSTEMS[name][mode][key])
