"""Reusable widgets for the production interface: buttons, cards, fields, swatches, toasts."""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, List, Sequence, Tuple

import customtkinter as ctk

from . import theme as T


# --------------------------------------------------------------------------
# text + icon primitives
# --------------------------------------------------------------------------

class Icon(ctk.CTkLabel):
    def __init__(self, master, name: str, size: int = 14, color=None, **kw):
        super().__init__(
            master,
            text=T.ICON.get(name, "?"),
            font=T.icon_font(size),
            text_color=color or T.c("dim"),
            **kw,
        )


class Label(ctk.CTkLabel):
    def __init__(self, master, text="", style="body", color=None, **kw):
        super().__init__(
            master,
            text=text,
            font=T.f(style),
            text_color=color or T.c("text"),
            **kw,
        )


class Divider(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, height=1, fg_color=T.c("border"), **kw)


# --------------------------------------------------------------------------
# buttons
# --------------------------------------------------------------------------

class Btn(ctk.CTkButton):
    """One button with four looks: primary, ghost, subtle, danger."""

    def __init__(self, master, text="", icon: str = None, variant="ghost",
                 command=None, height=34, **kw):
        label = text
        if icon:
            label = (T.ICON.get(icon, "") + "  " + text).strip()

        looks = {
            "primary": dict(
                fg_color=T.c("solid"), hover_color=T.c("solid_hi"),
                text_color=T.c("on_solid"), border_width=0,
            ),
            "accent": dict(
                fg_color=T.accent(), hover_color=T.accent_hover(),
                text_color=T.accent_on(), border_width=0,
            ),
            "ghost": dict(
                fg_color=T.c("card_alt"), hover_color=T.c("card_hi"),
                text_color=T.c("text"), border_width=1,
                border_color=T.c("border"),
            ),
            "subtle": dict(
                fg_color="transparent", hover_color=T.c("card_hi"),
                text_color=T.c("dim"), border_width=0,
            ),
            "danger": dict(
                fg_color="transparent",
                hover_color=T.mix(T.res(T.c("err")), T.res(T.c("surface")), 0.82),
                text_color=T.c("err"), border_width=1,
                border_color=T.c("border"),
            ),
        }[variant]
        looks.update(kw)

        # A glyph mixed with latin text still resolves: tk falls back per
        # character, so the UI font can carry both.
        font = T.icon_font(13) if (icon and not text) else T.f("strong")

        super().__init__(
            master, text=label, command=command, height=height,
            corner_radius=T.R_CTRL, font=font, **looks,
        )
        self.variant = variant


class RecentFileButton(Btn):
    """Fit filenames intentionally, retaining both ends and the extension."""
    def __init__(self, master, path, command):
        from pathlib import Path
        self.filename = Path(path).name
        super().__init__(master, self.filename, command=command, width=1,
                         anchor='w')
        self.configure(font=T.f('body'))
        Tooltip(self, str(path))
        self.bind('<Configure>', self._fit_name, add='+')

    def _fit_name(self, _event=None):
        budget = max(20, self.winfo_width() / self._get_widget_scaling() - 28)
        text = self.filename
        if self._font.measure(text) > budget:
            count = len(text) - 1
            while count > 2:
                left = (count + 1) // 2
                candidate = text[:left] + '…' + text[-(count - left):]
                if self._font.measure(candidate) <= budget:
                    text = candidate
                    break
                count -= 1
            else:
                text = '…'
        if self.cget('text') != text:
            self.configure(text=text)


class IconBtn(ctk.CTkButton):
    def __init__(self, master, icon: str, command=None, size=32, glyph=15,
                 tooltip: str = None, variant="subtle", **kw):
        looks = {
            "subtle": dict(fg_color="transparent", hover_color=T.c("card_hi"),
                           text_color=T.c("dim")),
            "ghost": dict(fg_color=T.c("card_alt"), hover_color=T.c("card_hi"),
                          text_color=T.c("text")),
            "accent": dict(fg_color=T.accent_soft(),
                           hover_color=T.accent_soft_hi(),
                           text_color=T.accent_text()),
        }[variant]
        looks.update(kw)
        super().__init__(
            master, text=T.ICON.get(icon, "?"), width=size, height=size,
            corner_radius=T.R_CTRL, font=T.icon_font(glyph), command=command,
            **looks,
        )
        if tooltip:
            Tooltip(self, tooltip)

    def set_icon(self, icon: str):
        self.configure(text=T.ICON.get(icon, "?"))


class Tooltip:
    """Small delayed tooltip; tkinter has none built in."""

    def __init__(self, widget, text: str, delay: int = 450):
        self.widget, self.text, self.delay = widget, text, delay
        self._after = None
        self._win = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None

    def _show(self):
        if self._win or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._win = tk.Toplevel(self.widget)
        self._win.wm_overrideredirect(True)
        self._win.attributes("-topmost", True)
        frame = ctk.CTkFrame(self._win, corner_radius=6, fg_color=T.c("card_hi"),
                             border_width=1, border_color=T.c("border_hi"))
        frame.pack()
        Label(frame, self.text, "small", T.c("text")).pack(padx=8, pady=3)
        self._win.update_idletasks()
        w = self._win.winfo_width()
        self._win.wm_geometry("+%d+%d" % (x - w // 2, y))

    def _hide(self, _e=None):
        self._cancel()
        if self._win:
            self._win.destroy()
            self._win = None


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

class Card(ctk.CTkFrame):
    """Titled surface. Content goes into ``.body``."""

    def __init__(self, master, title: str = None, icon: str = None,
                 hint: str = None, actions: Sequence[tuple] = (), **kw):
        super().__init__(master, corner_radius=T.R_CARD, fg_color=T.c("card"),
                         border_width=1, border_color=T.c("border"), **kw)
        self.grid_columnconfigure(0, weight=1)
        row = 0

        if title:
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.grid(row=0, column=0, sticky="ew", padx=13, pady=(10, 0))
            head.grid_columnconfigure(1, weight=1)
            col = 0
            if icon:
                # Decoration stays neutral; the accent is spent on actions.
                Icon(head, icon, 12, T.c("faint"), height=16).grid(row=0, column=0,
                                                        padx=(0, 7))
                col = 1
            Label(head, T.caps(title), "eyebrow", T.c("dim"), height=16).grid(
                row=0, column=col, sticky="w")
            for i, (ic, cb, tip) in enumerate(actions):
                IconBtn(head, ic, cb, size=24, glyph=12, tooltip=tip).grid(
                    row=0, column=2 + i, padx=(4, 0))
            Divider(self).grid(row=1, column=0, sticky="ew", padx=13,
                               pady=(9, 0))
            row = 2

        if hint:
            Label(self, hint, "small", T.c("faint")).grid(
                row=row, column=0, sticky="w", padx=13, pady=(9, 0))
            row += 1

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=row, column=0, sticky="nsew", padx=13, pady=(11, 12))
        self.body.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(row, weight=1)


class Chip(ctk.CTkFrame):
    """Compact pill for metadata (dimensions, color counts, ...)."""

    def __init__(self, master, text="", icon: str = None, tone="neutral", **kw):
        tones = {
            "neutral": (T.c("card_alt"), T.c("dim")),
            "accent": (T.accent_soft(), T.accent_text()),
            "ok": (T.c("card_alt"), T.c("ok")),
            "warn": (T.c("card_alt"), T.c("warn")),
        }
        bg, fg = tones[tone]
        super().__init__(master, corner_radius=T.R_PILL, fg_color=bg,
                         border_width=1, border_color=T.c("border"), **kw)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(padx=8, pady=2)
        if icon:
            Icon(inner, icon, 11, fg).pack(side="left", padx=(0, 5))
        self._label = Label(inner, text, "mono_s", fg)
        self._label.pack(side="left")

    def set(self, text: str):
        self._label.configure(text=text)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

class Field(ctk.CTkFrame):
    """Label above a control, with an optional right-aligned value readout."""

    def __init__(self, master, label: str, hint: str = None, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        Label(head, T.caps(label), "eyebrow", T.c("dim"), height=16).grid(
            row=0, column=0, sticky="w")
        self.value_label = Label(head, hint or "", "mono_s", T.c("faint"), height=16)
        self.value_label.grid(row=0, column=1, sticky="e")
        self.control = ctk.CTkFrame(self, fg_color="transparent")
        self.control.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.control.grid_columnconfigure(0, weight=1)


class SliderField(Field):
    """Slider plus a typed numeric entry that stay in sync."""

    def __init__(self, master, label, from_, to, value, step=None,
                 integer=True, command: Callable = None, typed_bounds=None, **kw):
        super().__init__(master, label, **kw)
        self._cmd = command
        self._from, self._to = from_, to
        self._integer = integer
        self._step = step if step is not None else (1 if integer else 0.01)
        self._typed_min, self._typed_max = typed_bounds or (from_, to)

        self.var = tk.DoubleVar(value=value)
        steps = int(round((to - from_) / step)) if step else None
        self.slider = ctk.CTkSlider(
            self.control, from_=from_, to=to, number_of_steps=steps,
            command=self._on_slide, height=16,
            # Neutral travel, accent only on the part you grab - the accent
            # has to stay scarce or it stops meaning "act here".
            fg_color=T.c("card_hi"), progress_color=T.c("border_hi"),
            button_color=T.accent(), button_hover_color=T.accent_hover(),
            button_length=0, button_corner_radius=8, corner_radius=2,
        )
        self.slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.entry = ctk.CTkEntry(
            self.control, width=64, height=28, corner_radius=T.R_CTRL,
            fg_color=T.c("card_alt"), border_color=T.c("border"),
            border_width=1, font=T.f("mono"), justify="center",
            text_color=T.c("text"),
        )
        self.entry.grid(row=0, column=1)
        self.entry.bind("<Return>", self._on_type)
        self.entry.bind("<FocusOut>", self._on_type)
        self.entry.bind("<Up>", lambda event: self._nudge(1))
        self.entry.bind("<Down>", lambda event: self._nudge(-1))
        if typed_bounds is not None:
            upper = "no fixed maximum" if self._typed_max is None else str(self._typed_max)
            Tooltip(self.entry, f"Type a value: {self._typed_min} to {upper}. "
                               f"Slider range: {from_} to {to}.")
        self._sync_entry()

    def _fmt(self, v):
        return str(int(round(v))) if self._integer else ("%.2f" % v)

    def _sync_entry(self):
        # The slider represents a convenient range, not the full typed domain.
        # Keeping its value separate prevents CTk from clamping typed values.
        self.slider.set(max(self._from, min(self._to, self.var.get())))
        self.entry.delete(0, "end")
        self.entry.insert(0, self._fmt(self.var.get()))

    def _on_slide(self, _v):
        self.var.set(_v)
        self._sync_entry()
        if self._cmd:
            self._cmd(self.get())

    def _read_entry(self):
        v = float(self.entry.get())
        if not math.isfinite(v):
            raise ValueError('Expected a finite number')
        if self._integer and not v.is_integer():
            raise ValueError('Expected a whole number')
        if self._typed_min is not None and v < self._typed_min:
            raise ValueError('Below the valid range')
        if self._typed_max is not None and v > self._typed_max:
            raise ValueError('Above the valid range')
        return v

    def _nudge(self, direction):
        if self.entry.cget('state') == 'disabled':
            return 'break'
        try:
            v = self._read_entry()
        except ValueError:
            v = self.var.get()
        v = round(v + direction * self._step, 12)
        if self._typed_min is not None:
            v = max(self._typed_min, v)
        if self._typed_max is not None:
            v = min(self._typed_max, v)
        changed = v != self.var.get()
        self.var.set(v)
        self._sync_entry()
        if changed and self._cmd:
            self._cmd(self.get())
        return 'break'

    def _on_type(self, _e=None):
        try:
            v = self._read_entry()
        except ValueError:
            self._sync_entry()
            return
        changed = v != self.var.get()
        self.var.set(v)
        self._sync_entry()
        if changed and self._cmd:
            self._cmd(self.get())

    def get(self):
        v = self.var.get()
        return int(round(v)) if self._integer else v


class Segmented(ctk.CTkFrame):
    """Segmented control - the modern stand-in for a row of radio buttons."""

    def __init__(self, master, values: Sequence[str], value: str = None,
                 icons: Sequence[str] = None, command: Callable = None,
                 height=34, seg_width=1, **kw):
        super().__init__(master, corner_radius=T.R_CTRL,
                         fg_color=T.c("card_alt"), border_width=1,
                         border_color=T.c("border"), **kw)
        self._cmd = command
        self._buttons = {}
        self._value = value or values[0]
        for i, v in enumerate(values):
            self.grid_columnconfigure(i, weight=1)
            text = v
            if icons and i < len(icons) and icons[i]:
                text = T.ICON.get(icons[i], "") + "  " + v
            b = ctk.CTkButton(
                self, text=text, height=height - 8, corner_radius=T.R_CTRL - 2,
                # CTkButton asks for 140px by default; in an equal-weight grid
                # that request fights the allocation and starves later columns.
                width=seg_width, font=T.f("body"),
                command=lambda x=v: self.set(x, notify=True),
                fg_color="transparent", hover_color=T.c("card_hi"),
                text_color=T.c("dim"),
            )
            b.grid(row=0, column=i, sticky="ew", padx=3, pady=4)
            self._buttons[v] = b
        self._paint()

    def _paint(self):
        for v, b in self._buttons.items():
            if v == self._value:
                b.configure(fg_color=T.accent_soft(),
                            text_color=T.accent_text(),
                            hover_color=T.accent_soft_hi(), font=T.f("strong"))
            else:
                b.configure(fg_color="transparent", text_color=T.c("dim"),
                            hover_color=T.c("card_hi"), font=T.f("body"))

    def set(self, value: str, notify=False):
        if value not in self._buttons:
            return
        self._value = value
        self._paint()
        if notify and self._cmd:
            self._cmd(value)

    def get(self):
        return self._value


class Toggle(ctk.CTkFrame):
    """Switch with a title and an explanatory second line."""

    def __init__(self, master, text: str, hint: str = None, value=False,
                 command: Callable = None, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        Label(self, text, "body", height=18).grid(row=0, column=0, sticky="w")
        if hint:
            Label(self, hint, "tiny", T.c("dim"), justify="left", height=15).grid(
                row=1, column=0, sticky="w", pady=(1, 0))
        self.var = tk.BooleanVar(value=value)
        self.switch = ctk.CTkSwitch(
            self, text="", variable=self.var, width=40,
            switch_width=38, switch_height=18, corner_radius=10,
            fg_color=("#8B8B86", "#696963"), progress_color=T.accent(),
            button_color=("#FFFFFF", "#E7EBF2"),
            button_hover_color=("#FFFFFF", "#FFFFFF"),
            command=command,
        )
        self.switch.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

    def get(self):
        return self.var.get()


def make_entry(master, value="", width=None, placeholder=None, **kw):
    e = ctk.CTkEntry(
        master, height=32, corner_radius=T.R_CTRL, fg_color=T.c("card_alt"),
        border_color=T.c("border"), border_width=1, font=T.f("body"),
        text_color=T.c("text"), placeholder_text=placeholder, **kw)
    if width:
        e.configure(width=width)
    if value:
        e.insert(0, value)
    return e


def make_menu(master, values, variable=None, command=None, width=140):
    return ctk.CTkOptionMenu(
        master, values=list(values), variable=variable, command=command,
        width=width, height=32, corner_radius=T.R_CTRL,
        fg_color=T.c("card_alt"), button_color=T.c("card_alt"),
        button_hover_color=T.c("card_hi"), text_color=T.c("text"),
        dropdown_fg_color=T.c("card"), dropdown_hover_color=T.c("card_hi"),
        dropdown_text_color=T.c("text"), font=T.f("body"),
        dropdown_font=T.f("body"),
    )


# --------------------------------------------------------------------------
# palette widgets
# --------------------------------------------------------------------------

def hexc(col) -> str:
    if isinstance(col, str):
        return col
    return "#%02x%02x%02x" % tuple(int(v) for v in col[:3])


class SwatchStrip(tk.Canvas):
    """Continuous rounded bar of palette colors."""

    def __init__(self, master, colors: Sequence[Tuple[int, int, int]],
                 width=200, height=18, radius=5, bg_token=None, **kw):
        self._bg_token = bg_token or T.c("card")
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, bd=0,
                         bg=T.res(self._bg_token), **kw)
        self._colors = list(colors)
        self._radius = radius
        self.bind("<Configure>", lambda e: self.redraw())
        self.redraw()

    def set_colors(self, colors):
        self._colors = list(colors)
        self.redraw()

    def set_bg_token(self, token):
        self._bg_token = token
        self.redraw()

    def redraw(self):
        self.delete("all")
        bg = T.res(self._bg_token)
        self.configure(bg=bg)
        w = self.winfo_width() or int(self["width"])
        h = self.winfo_height() or int(self["height"])
        n = max(1, len(self._colors))
        for i, col in enumerate(self._colors):
            x0 = i * w / n
            x1 = (i + 1) * w / n
            self.create_rectangle(x0, 0, x1 + 1, h, fill=hexc(col), outline="")
        # Punch the four corners back out to the parent color so the row of
        # hard-edged slices reads as one rounded bar.
        r = self._radius
        for cx, cy, sx, sy in ((0, 0, 1, 1), (w, 0, -1, 1),
                               (0, h, 1, -1), (w, h, -1, -1)):
            pts = [cx, cy]
            for k in range(9):
                a = math.pi / 2 * k / 8
                pts += [cx + sx * r * (1 - math.sin(a)),
                        cy + sy * r * (1 - math.cos(a))]
            self.create_polygon(pts, fill=bg, outline="")


class PaletteCard(ctk.CTkFrame):
    """Selectable palette row: name, count, source badge, swatches."""

    def __init__(self, master, name: str, colors, badge: str = None,
                 command: Callable = None, placeholder='No colors', **kw):
        super().__init__(master, corner_radius=T.R_CARD,
                         fg_color=T.c("card_alt"), border_width=1,
                         border_color=T.c("border"), **kw)
        self.name = name
        self._cmd = command
        self.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))
        head.grid_columnconfigure(0, weight=1)
        Label(head, name, "strong", height=18).grid(row=0, column=0, sticky="w")
        meta = "%d colors" % len(colors) if colors else placeholder
        if badge:
            meta = badge + " · " + meta
        Label(head, meta, "tiny", T.c("dim"), height=14).grid(row=1, column=0,
                                                     sticky="w", pady=(1, 0))

        self.strip = SwatchStrip(self, colors, height=16,
                                 bg_token=T.c("card_alt"))
        self.strip.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 7))

        self._check = Icon(self, "check_c", 15, T.accent_text())

        self.bind("<Button-1>", self._click)
        for w in (head, self.strip):
            w.bind("<Button-1>", self._click)
        for child in head.winfo_children():
            child.bind("<Button-1>", self._click)
        self._hover_bind()

    def _hover_bind(self):
        def enter(_e):
            if not self._selected:
                self.configure(border_color=T.c("border_hi"))

        def leave(_e):
            if not self._selected:
                self.configure(border_color=T.c("border"))

        self._selected = False
        self.bind("<Enter>", enter, add="+")
        self.bind("<Leave>", leave, add="+")

    def _click(self, _e=None):
        if self._cmd:
            self._cmd(self.name)

    def set_selected(self, on: bool):
        self._selected = on
        token = T.accent_soft() if on else T.c("card_alt")
        self.configure(border_color=T.accent() if on else T.c("border"),
                       fg_color=token)
        self.strip.set_bg_token(token)
        if on:
            self._check.place(relx=1.0, rely=0.0, x=-9, y=9, anchor="ne")
        else:
            self._check.place_forget()


# --------------------------------------------------------------------------
# status + toast
# --------------------------------------------------------------------------

class StatusBar(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, height=T.STATUS_H, corner_radius=0,
                         fg_color=T.c("surface"), **kw)
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)

        self._dot = tk.Canvas(self, width=8, height=8, highlightthickness=0,
                              bd=0, bg=T.res(T.c("surface")))
        self._dot.grid(row=0, column=0, padx=(14, 8))
        self._dot_id = self._dot.create_rectangle(
            1, 1, 7, 7, fill=T.res(T.c("ok")), outline="")

        self._text = Label(self, "Ready", "small", T.c("dim"))
        self._text.grid(row=0, column=1, sticky="w")

        self.right = ctk.CTkFrame(self, fg_color="transparent")
        self.right.grid(row=0, column=2, sticky="e", padx=(0, 12))

        self.progress = ctk.CTkProgressBar(
            self, width=130, height=5, corner_radius=3,
            fg_color=T.c("card_hi"), progress_color=T.accent())
        self._progress_shown = False

        self._spin_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spin_i = 0
        self._spin_after = None
        self._spin_text = ""

    def set(self, text: str, tone="idle", busy=False):
        tones = {"idle": T.c("ok"), "busy": T.accent(),
                 "warn": T.c("warn"), "err": T.c("err")}
        self._dot.configure(bg=T.res(T.c("surface")))
        self._dot.itemconfigure(self._dot_id,
                                fill=T.res(tones.get(tone, T.c("ok"))))
        self._stop_spin()
        if busy:
            self._spin_text = text
            self._spin_i = 0
            self._tick()
        else:
            self._text.configure(text=text)

    def _tick(self):
        ch = self._spin_chars[self._spin_i % len(self._spin_chars)]
        self._text.configure(text=ch + "  " + self._spin_text)
        self._spin_i += 1
        self._spin_after = self.after(80, self._tick)

    def _stop_spin(self):
        if self._spin_after:
            try:
                self.after_cancel(self._spin_after)
            except tk.TclError:
                pass
            self._spin_after = None

    def destroy(self):
        self._stop_spin()
        super().destroy()

    def show_progress(self, value: float = 0.0):
        if not self._progress_shown:
            self.progress.grid(row=0, column=3, padx=(10, 14))
            self._progress_shown = True
        self.progress.set(value)

    def hide_progress(self):
        if self._progress_shown:
            self.progress.grid_forget()
            self._progress_shown = False


class Toast(ctk.CTkFrame):
    """Transient bottom-right notification, stacked by the host."""

    def __init__(self, master, text: str, icon="info", tone="neutral",
                 on_done: Callable = None):
        super().__init__(master, corner_radius=T.R_CARD, fg_color=T.c("card_hi"),
                         border_width=1, border_color=T.c("border_hi"))
        tones = {"neutral": T.c("dim"), "ok": T.c("ok"), "warn": T.c("warn"),
                 "err": T.c("err"), "accent": T.accent_text()}
        fg = tones.get(tone, T.c("dim"))
        Icon(self, icon, 14, fg).grid(row=0, column=0, padx=(12, 8), pady=10)
        Label(self, text, "small", T.c("text")).grid(row=0, column=1,
                                                     padx=(0, 14), pady=10)
        self._on_done = on_done
        self._dismiss_id = self.after(2800, self.dismiss)
        self.bind("<Button-1>", lambda e: self.dismiss())

    def dismiss(self):
        if self._dismiss_id is not None:
            self.after_cancel(self._dismiss_id)
            self._dismiss_id = None
        if not self.winfo_exists():
            return
        if self._on_done:
            self._on_done(self)
        self.destroy()


class ToastHost:
    """Owns toast stacking inside a parent widget."""

    def __init__(self, parent):
        self.parent = parent
        self._toasts: List[Toast] = []

    def show(self, text, icon="info", tone="neutral"):
        t = Toast(self.parent, text, icon, tone, on_done=self._remove)
        self._toasts.append(t)
        self._layout()

    def _remove(self, toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
        self._layout()

    def _layout(self):
        y = -16
        for t in reversed(self._toasts):
            if t.winfo_exists():
                t.place(relx=1.0, rely=1.0, x=-16, y=y, anchor="se")
                t.update_idletasks()
                y -= t.winfo_reqheight() + 8
