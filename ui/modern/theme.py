"""Design tokens for the Dither Pie GUI mockup.

Everything visual comes from here: colors, type scale, spacing and the icon
glyph table.  Colors are (light, dark) pairs so CustomTkinter can swap them
when the appearance mode changes; ``res()`` resolves a pair down to a single
hex string for raw tkinter canvases that cannot take pairs.

Palettes come from designdotmd.directory, which catalogs DESIGN.md token sets
with the reasoning attached. Each entry there gives one mode; the full light
and dark ramps here are built from four anchors per mode (paper/base, ink,
muted, accent) so the two modes stay siblings rather than separate guesses.
"""

from __future__ import annotations

import tkinter.font as tkfont
import customtkinter as ctk


# --------------------------------------------------------------------------
# color math (needed by the ramp builder below)
# --------------------------------------------------------------------------

def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(t) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v)))) for v in t)


def mix(a: str, b: str, t: float) -> str:
    """Blend ``a`` toward ``b`` by ``t`` (0 = a, 1 = b)."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(ra[i] + (rb[i] - ra[i]) * t for i in range(3)))


def shade(c: str, amount: float) -> str:
    """Lighten (positive) or darken (negative) a color."""
    target = "#FFFFFF" if amount > 0 else "#000000"
    return mix(c, target, abs(amount))


# --------------------------------------------------------------------------
# design systems
# --------------------------------------------------------------------------

# Each system supplies, per mode, four anchors plus the accent trio:
#   paper/base - the page; ink - body text; muted - secondary text
#   accent     - the fill color for the one thing you act on
#   accent_text- the same hue pushed to a readable weight for text
#   accent_on  - what sits legibly on top of an accent fill
SYSTEMS = {
    "Teenage Engineering": {
        "note": "Bone-white plastic, black knobs, one orange dot.",
        "light": dict(paper="#F4F3EF", ink="#111111", muted="#6E6C68",
                      accent="#FF5A1F", accent_text="#D2440F",
                      accent_on="#FFFFFF"),
        "dark": dict(base="#121211", ink="#E9E7E2", muted="#8B8881",
                     accent="#FF5A1F", accent_text="#FF7A45",
                     accent_on="#141312"),
    },
    "Film Photography": {
        "note": "35mm grain: gelatin silver, Kodak yellow, frame numbers.",
        "light": dict(paper="#FBF7EC", ink="#151412", muted="#777268",
                      accent="#E0AE24", accent_text="#8A6510",
                      accent_on="#151412"),
        "dark": dict(base="#131210", ink="#EEEAE0", muted="#877F73",
                     accent="#F2C94C", accent_text="#F2C94C",
                     accent_on="#151412"),
    },
    "Zine Risograph": {
        "note": "Riso-printed zines: fluoro pink on pulp paper.",
        "light": dict(paper="#F8F1DF", ink="#181818", muted="#6B665C",
                      accent="#FF48B0", accent_text="#C4187E",
                      accent_on="#FFFFFF"),
        "dark": dict(base="#141311", ink="#F0E8D6", muted="#847D70",
                     accent="#FF48B0", accent_text="#FF74C4",
                     accent_on="#141311"),
    },
    "Zed Editor": {
        "note": "Editor-dark, warmer than pitch, glyph-sharp.",
        "light": dict(paper="#F2F1ED", ink="#1D1D1B", muted="#6E6D67",
                      accent="#2E7E9E", accent_text="#1E6A88",
                      accent_on="#FFFFFF"),
        "dark": dict(base="#161614", ink="#E5E4DF", muted="#7B7A74",
                     accent="#5FB5D6", accent_text="#7FC8E4",
                     accent_on="#101514"),
    },
    "Swiss Grid": {
        "note": "Helvetica, a twelve-column grid, and a single red.",
        "light": dict(paper="#FFFFFF", ink="#000000", muted="#555555",
                      accent="#DC2626", accent_text="#C81E1E",
                      accent_on="#FFFFFF"),
        "dark": dict(base="#0B0B0B", ink="#F2F2F2", muted="#8A8A8A",
                     accent="#DC2626", accent_text="#FF5F5A",
                     accent_on="#FFFFFF"),
    },
}

DEFAULT_SYSTEM = "Teenage Engineering"


def _build(system: dict) -> dict:
    """Expand four anchors per mode into the full token ramp."""
    lt, dk = system["light"], system["dark"]
    paper, ink_l, muted_l = lt["paper"], lt["ink"], lt["muted"]
    base, ink_d, muted_d = dk["base"], dk["ink"], dk["muted"]

    def pair(light_v, dark_v):
        return (light_v, dark_v)

    return {
        # The page itself sits a step below the panels so cards read as
        # objects laid on it rather than holes cut into it.
        "bg":        pair(mix(paper, ink_l, 0.06), base),
        "surface":   pair(paper, mix(base, ink_d, 0.035)),
        "card":      pair(mix(paper, "#FFFFFF", 0.55), mix(base, ink_d, 0.065)),
        "card_alt":  pair(mix(paper, ink_l, 0.045), mix(base, ink_d, 0.035)),
        "card_hi":   pair(mix(paper, ink_l, 0.10), mix(base, ink_d, 0.13)),
        "border":    pair(mix(paper, ink_l, 0.14), mix(base, ink_d, 0.105)),
        "border_hi": pair(mix(paper, ink_l, 0.26), mix(base, ink_d, 0.22)),
        "text":      pair(ink_l, ink_d),
        "dim":       pair(muted_l, muted_d),
        "faint":     pair(mix(muted_l, paper, 0.34), mix(muted_d, base, 0.30)),
        "canvas":    pair(mix(paper, ink_l, 0.13), mix(base, "#000000", 0.45)),
        # Status hues are muted toward the system's own warmth: saturated
        # signal greens and pinks are what make an interface look generated.
        "ok":        pair("#2F7A3E", "#7FBF8C"),
        "warn":      pair("#8A6210", "#D9A441"),
        "err":       pair("#A6322B", "#E0776D"),
        # The main action is ink on paper (or paper on ink): high contrast
        # without spending the accent, which then still means "this one".
        "solid":     pair(ink_l, ink_d),
        "solid_hi":  pair(mix(ink_l, paper, 0.18), mix(ink_d, base, 0.16)),
        "on_solid":  pair(paper, base),
        "accent":      pair(lt["accent"], dk["accent"]),
        "accent_text": pair(lt["accent_text"], dk["accent_text"]),
        "accent_on":   pair(lt["accent_on"], dk["accent_on"]),
    }


COL: dict = {}
_state = {"system": DEFAULT_SYSTEM}


def set_system(name: str) -> None:
    if name not in SYSTEMS:
        return
    _state["system"] = name
    COL.clear()
    COL.update(_build(SYSTEMS[name]))


def system_name() -> str:
    return _state["system"]


def system_note(name: str = None) -> str:
    return SYSTEMS[name or _state["system"]]["note"]


def c(name: str) -> tuple:
    return COL[name]


def accent() -> tuple:
    """Fill color. Reserved for the one thing the user is about to act on."""
    return COL["accent"]


def accent_text() -> tuple:
    """Accent as type: same hue, pushed to a weight that stays readable."""
    return COL["accent_text"]


def accent_on() -> tuple:
    return COL["accent_on"]


def accent_hover() -> tuple:
    lo, hi = COL["accent"]
    return (shade(lo, -0.12), shade(hi, 0.10))


def accent_soft() -> tuple:
    """Accent tinted into the panel background - used for selected states."""
    lo, hi = COL["accent"]
    return (mix(lo, COL["surface"][0], 0.91), mix(hi, COL["surface"][1], 0.90))


def accent_soft_hi() -> tuple:
    lo, hi = COL["accent"]
    return (mix(lo, COL["surface"][0], 0.84), mix(hi, COL["surface"][1], 0.83))


def token_pairs() -> list:
    """Every (light, dark) pair the system defines, for live re-theming."""
    pairs = list(COL.values())
    pairs += [accent_hover(), accent_soft(), accent_soft_hi()]
    return pairs


set_system(DEFAULT_SYSTEM)


# Kept so existing call sites and the checks keep working.
ACCENTS = SYSTEMS
set_accent = set_system
accent_name = system_name


def is_dark() -> bool:
    return ctk.get_appearance_mode().lower() == "dark"


def res(token) -> str:
    """Resolve a (light, dark) pair - or a plain string - to one hex color."""
    if isinstance(token, (tuple, list)):
        return token[1] if is_dark() else token[0]
    return token


# --------------------------------------------------------------------------
# type
# --------------------------------------------------------------------------

UI = "Segoe UI Variable Text"
UI_DISPLAY = "Segoe UI Variable Display"
MONO = "Cascadia Mono"
ICON_FONT = "Segoe Fluent Icons"

FONTS: dict = {}


def init_fonts() -> None:
    """Build the type scale. Must run after a Tk root exists."""
    families = set(tkfont.families())
    ui = UI if UI in families else "Segoe UI"
    disp = UI_DISPLAY if UI_DISPLAY in families else ui
    mono = MONO if MONO in families else "Consolas"

    FONTS["_icon_family"] = ICON_FONT if ICON_FONT in families else "Segoe MDL2 Assets"
    FONTS.update({
        "display": ctk.CTkFont(disp, 17, "bold"),
        "h1":      ctk.CTkFont(disp, 15, "bold"),
        "h2":      ctk.CTkFont(ui, 12, "bold"),
        # Engraved-label type: small mono caps, the way a panel legend is set.
        "eyebrow": ctk.CTkFont(mono, 10),
        "label":   ctk.CTkFont(ui, 12),
        "body":    ctk.CTkFont(ui, 12),
        "strong":  ctk.CTkFont(ui, 12, "bold"),
        "small":   ctk.CTkFont(ui, 11),
        "tiny":    ctk.CTkFont(ui, 10),
        "tiny_b":  ctk.CTkFont(ui, 10, "bold"),
        "mono":    ctk.CTkFont(mono, 11),
        "mono_s":  ctk.CTkFont(mono, 10),
    })
    FONTS["_raw_ui"] = ui
    FONTS["_raw_mono"] = mono


def f(name: str):
    return FONTS[name]


def icon_font(size: int = 14, bold: bool = False):
    fam = FONTS.get("_icon_family", "Segoe MDL2 Assets")
    return ctk.CTkFont(fam, size, "bold" if bold else "normal")


def caps(text: str) -> str:
    """Uppercase with hair spacing - tk has no letter-spacing of its own."""
    return " ".join(text.upper())


# --------------------------------------------------------------------------
# icons (Segoe Fluent Icons private-use codepoints)
# --------------------------------------------------------------------------

def _g(code: str) -> str:
    return chr(int(code, 16))


ICON = {
    "image":     _g("E91B"),
    "video":     _g("E714"),
    "open":      _g("E8E5"),
    "folder":    _g("E8B7"),
    "save":      _g("E74E"),
    "export":    _g("E7B5"),
    "settings":  _g("E713"),
    "sliders":   _g("E9E9"),
    "palette":   _g("E2B1"),
    "grid":      _g("F0E2"),
    "grid_fine": _g("E80A"),
    "layers":    _g("F156"),
    "crop":      _g("E7A8"),
    "dropper":   _g("EF3C"),
    "pencil":    _g("E70F"),
    "wand":      _g("E945"),
    "undo":      _g("E7A7"),
    "redo":      _g("E7A6"),
    "zoom_in":   _g("E8A3"),
    "zoom_out":  _g("E71F"),
    "fit":       _g("E9A6"),
    "expand":    _g("E740"),
    "eye":       _g("E890"),
    "check":     _g("E8FB"),
    "check_c":   _g("E930"),
    "close":     _g("E711"),
    "plus":      _g("E710"),
    "trash":     _g("E74D"),
    "search":    _g("E721"),
    "info":      _g("E946"),
    "warn":      _g("E7BA"),
    "help":      _g("E9CE"),
    "refresh":   _g("E72C"),
    "rotate":    _g("E9F3"),
    "play":      _g("E768"),
    "download":  _g("E896"),
    "link":      _g("E71B"),
    "sun":       _g("E706"),
    "moon":      _g("E708"),
    "contrast":  _g("E793"),
    "chev_r":    _g("E76C"),
    "chev_d":    _g("E70D"),
    "list":      _g("E71D"),
    "star":      _g("E734"),
    "timer":     _g("E916"),
    "monitor":   _g("E7F4"),
    "copy":      _g("E8C8"),
    "shuffle":   _g("E8B3"),
    "arrow_ur":  _g("E8AD"),
    # Window chrome, for the app's own title bar.
    "win_min":     _g("E921"),
    "win_max":     _g("E922"),
    "win_restore": _g("E923"),
    "win_close":   _g("E8BB"),
}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

# Every catalogued system in this family lands on 2-6px corners. Large radii
# plus tinted cards everywhere is the look being avoided here.
R_CARD = 4
R_CTRL = 3
R_PILL = 3

PAD = 12
GAP = 8

RAIL_W = 68
PANEL_W = 328
TOPBAR_H = 40
STATUS_H = 30


def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle for raw tk canvases (smooth polygon approximation)."""
    r = min(r, abs(x2 - x1) / 2, abs(y2 - y1) / 2)
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)
