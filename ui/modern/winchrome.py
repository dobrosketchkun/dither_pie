"""Hide the native Windows caption while keeping a real, normal window.

There are three ways to get rid of the title bar in tkinter:

1. ``overrideredirect(True)`` removes the entire frame. It is the one every
   tutorial reaches for, and it also removes the taskbar button, Aero Snap,
   Win+Arrow, the drop shadow and the resize borders - all of which then have
   to be reimplemented badly.
2. Recolor the native bar instead (``DwmSetWindowAttribute`` with
   ``DWMWA_CAPTION_COLOR``, Windows 11 only). Keeps every native behaviour and
   costs nothing, but the bar is still a Windows bar.
3. Strip ``WS_CAPTION`` from the window style and keep ``WS_THICKFRAME`` and
   the min/max boxes. The caption is gone, but the window is still a normal
   top-level window: taskbar button, snapping, resize borders and Alt+Tab all
   keep working.

This module does (3), with (2) available as ``tint_caption`` for a fallback.

Two things then have to be reimplemented, and both are driven from
``work_area`` rather than by Windows:

* **Maximize.** With no caption, Windows sizes a maximized window to the whole
  monitor instead of the work area, so it covers the taskbar - measured here
  as 1080px against a 1032px work area.
* **Moving the window.** The usual trick is ``ReleaseCapture()`` followed by
  ``WM_NCLBUTTONDOWN``/``HTCAPTION``, which hands the drag to the OS and keeps
  Aero Snap for free. It cannot be used from Python: that call blocks inside
  Windows' modal move loop while the loop re-enters Tk and calls back into the
  interpreter, and the interpreter dies with "PyEval_RestoreThread: the
  function must be called with the GIL held". ``ctypes.PyDLL`` (GIL held) and
  ``PostMessage`` (non-blocking) were both tried; the first crashes the same
  way once a second thread is running, the second never reaches
  ``DefWindowProc`` and the window simply does not move. So the app moves the
  window itself and implements edge snapping on top of ``work_area``.

Every call is a no-op returning False off Windows, so the app keeps running.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

IS_WINDOWS = sys.platform == "win32"

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
WS_EX_APPWINDOW = 0x00040000

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SWP_NOACTIVATE = 0x0010

DWMWA_CAPTION_COLOR = 35
DWMWA_WINDOW_CORNER_PREFERENCE = 33
MONITOR_DEFAULTTONEAREST = 2

# Two handles on purpose. ``windll`` releases the GIL for the duration of the
# call, which is right for calls that just return a value. It is fatal for any
# call that synchronously re-enters this window's procedure, because Tk can
# then call back into Python with no thread state - the interpreter aborts with
# "PyEval_RestoreThread: the GIL is released". ``PyDLL`` keeps the GIL held, so
# re-entrant callbacks are safe; it is only used for calls that return quickly.
_user32 = ctypes.windll.user32 if IS_WINDOWS else None
_user32_gil = ctypes.PyDLL("user32", use_last_error=False) if IS_WINDOWS else None
_dwmapi = ctypes.PyDLL("dwmapi", use_last_error=False) if IS_WINDOWS else None


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _RECT),
                ("rcWork", _RECT), ("dwFlags", ctypes.c_ulong)]


if IS_WINDOWS:
    # ctypes defaults to 32-bit integers, including for 64-bit handles.
    for dll in (_user32, _user32_gil):
        dll.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, wintypes.UINT]
        dll.SetWindowPos.restype = wintypes.BOOL
    _user32.GetParent.argtypes = [wintypes.HWND]
    _user32.GetParent.restype = wintypes.HWND
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL
    _user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.MonitorFromWindow.restype = wintypes.HANDLE
    _user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MONITORINFO)]
    _user32.GetMonitorInfoW.restype = wintypes.BOOL


class StableTitlebar:
    """Avoid CTk's withdraw/update cycle when changing Windows appearance.

    Withdrawing an owner also hides its transient dialogs. CTk can then save
    'withdrawn' as a dialog's state and leave it hidden with an active grab.
    DWM can update the caption directly without hiding either window.
    """

    def _windows_set_titlebar_color(self, color_mode):
        if not IS_WINDOWS:
            return
        value = ctypes.c_int(color_mode.lower() == 'dark')
        hwnd = hwnd_of(self)
        if hwnd:
            for attribute in (20, 19):
                if _dwmapi.DwmSetWindowAttribute(
                        wintypes.HWND(hwnd), attribute, ctypes.byref(value),
                        ctypes.sizeof(value)) == 0:
                    break


def _long_funcs():
    """64-bit builds need the Ptr variants or the style is truncated."""
    try:
        get = _user32.GetWindowLongPtrW
        setter = _user32_gil.SetWindowLongPtrW
        get.restype = ctypes.c_longlong
        get.argtypes = [wintypes.HWND, ctypes.c_int]
        setter.restype = ctypes.c_longlong
        setter.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
        return get, setter
    except AttributeError:
        return _user32.GetWindowLongW, _user32_gil.SetWindowLongW


def hwnd_of(window) -> int:
    """The real top-level HWND: Tk's own id is a child of the frame window."""
    if not IS_WINDOWS:
        return 0
    try:
        return _user32.GetParent(window.winfo_id()) or window.winfo_id()
    except Exception:
        return 0


def _apply_frame_change(hwnd) -> None:
    # Dispatches WM_NCCALCSIZE and friends to the window synchronously, so it
    # has to run with the GIL held.
    _user32_gil.SetWindowPos(wintypes.HWND(hwnd), 0, 0, 0, 0, 0,
                             SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE
                             | SWP_NOZORDER)


def hide_caption(window, resizable: bool = True) -> bool:
    """Drop the title bar; keep the taskbar button, snapping and resizing."""
    if not IS_WINDOWS:
        return False
    hwnd = hwnd_of(window)
    if not hwnd:
        return False
    try:
        get, setter = _long_funcs()
        style = get(hwnd, GWL_STYLE)
        style &= ~WS_CAPTION
        style |= WS_SYSMENU
        if resizable:
            style |= WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX
        else:
            style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX)
        if window.transient():
            style &= ~(WS_MAXIMIZEBOX | WS_MINIMIZEBOX)
        setter(hwnd, GWL_STYLE, style)
        # Without this a caption-less tool window can lose its taskbar button.
        if not window.transient():
            setter(hwnd, GWL_EXSTYLE, get(hwnd, GWL_EXSTYLE) | WS_EX_APPWINDOW)
        _apply_frame_change(hwnd)
        return True
    except Exception:
        return False


def restore_caption(window) -> bool:
    if not IS_WINDOWS:
        return False
    hwnd = hwnd_of(window)
    if not hwnd:
        return False
    try:
        get, setter = _long_funcs()
        setter(hwnd, GWL_STYLE, get(hwnd, GWL_STYLE) | WS_CAPTION | WS_SYSMENU
               | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_THICKFRAME)
        _apply_frame_change(hwnd)
        return True
    except Exception:
        return False


def round_corners(window, on: bool = True) -> bool:
    """Windows 11 corner preference: 2 = round, 1 = square."""
    if not IS_WINDOWS or _dwmapi is None:
        return False
    try:
        pref = ctypes.c_int(2 if on else 1)
        return _dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd_of(window)), DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(pref), ctypes.sizeof(pref)) == 0
    except Exception:
        return False


def tint_caption(window, hex_color: str) -> bool:
    """Fallback: keep the native bar but paint it the theme color (Win 11)."""
    if not IS_WINDOWS or _dwmapi is None:
        return False
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        colorref = ctypes.c_int((b << 16) | (g << 8) | r)
        return _dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd_of(window)), DWMWA_CAPTION_COLOR,
            ctypes.byref(colorref), ctypes.sizeof(colorref)) == 0
    except Exception:
        return False


def window_rect(window):
    """Outer window rect in screen coordinates, as (left, top, right, bottom).

    ``winfo_x``/``winfo_y`` report the client origin and CustomTkinter scales
    whatever goes through ``geometry()``, so neither is a safe basis for
    dragging.
    """
    if not IS_WINDOWS:
        return None
    hwnd = hwnd_of(window)
    if not hwnd:
        return None
    try:
        r = _RECT()
        if not _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r)):
            return None
        return r.left, r.top, r.right, r.bottom
    except Exception:
        return None


def move_window(window, x: int, y: int) -> bool:
    """Move without touching size. Goes around Tk and CustomTkinter entirely:
    a position-only ``geometry()`` makes Tk fall back to the requested size,
    and CustomTkinter's scaling makes a get/set round trip drift."""
    if not IS_WINDOWS:
        return False
    hwnd = hwnd_of(window)
    if not hwnd:
        return False
    try:
        # Dispatches WM_WINDOWPOSCHANGED synchronously, so GIL held.
        return bool(_user32_gil.SetWindowPos(
            wintypes.HWND(hwnd), 0, int(x), int(y), 0, 0,
            SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE))
    except Exception:
        return False


def set_window_rect(window, rect) -> bool:
    """Set outer bounds in physical screen pixels, including negative origins."""
    if not IS_WINDOWS:
        return False
    try:
        x, y, right, bottom = rect
        return bool(_user32_gil.SetWindowPos(
            hwnd_of(window), 0, int(x), int(y), int(right - x), int(bottom - y),
            SWP_NOZORDER | SWP_NOACTIVATE))
    except Exception:
        return False


def work_area(window):
    """Monitor work area (screen minus taskbar) as (x, y, width, height)."""
    if not IS_WINDOWS:
        return None
    try:
        monitor = _user32.MonitorFromWindow(hwnd_of(window),
                                            MONITOR_DEFAULTTONEAREST)
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if not _user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        r = info.rcWork
        return r.left, r.top, r.right - r.left, r.bottom - r.top
    except Exception:
        return None
