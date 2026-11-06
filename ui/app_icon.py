"""The Bayer logo used by the native window and Windows taskbar."""
import ctypes
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ICON_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'dither_pie.ico'
BAYER_MARK = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))


def icon_image(size):
    image = Image.new('RGBA', (size, size), '#f5f3ee')
    draw = ImageDraw.Draw(image)
    step, pad = size * 3 / 16, size / 8
    for y in range(4):
        for x in range(4):
            if (1 - (x + .5) / 4) * 16 > BAYER_MARK[y][x]:
                draw.rectangle((round(pad + x * step), round(pad + y * step),
                                round(pad + (x + 1) * step) - 1, round(pad + (y + 1) * step) - 1),
                               fill='#ff5c1a' if x + y == 3 else '#1a1a1a')
    return image


def set_app_identity():
    if sys.platform == 'win32':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('DitherPie.Desktop')


def set_app_icon(window):
    if sys.platform == 'win32':
        # iconbitmap also prevents CTk's delayed default icon from replacing it.
        window.iconbitmap(default=str(ICON_PATH))
        window.iconbitmap(str(ICON_PATH))
    else:
        from PIL import ImageTk
        window._app_icon = ImageTk.PhotoImage(icon_image(64), master=window)
        window.iconphoto(True, window._app_icon)
