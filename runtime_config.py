"""Compatibility for scientific-library CPU detection on current Windows."""
import ctypes
import sys
from ctypes import wintypes


def windows_physical_cpu_count():
    """Count processor-core records through Windows, without the removed WMIC."""
    if sys.platform != 'win32':
        return None
    query = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
    query.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    query.restype = wintypes.BOOL
    size = wintypes.DWORD()
    query(0, None, ctypes.byref(size))  # RelationProcessorCore
    if not size.value:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if not query(0, buffer, ctypes.byref(size)):
        return None
    offset, count = 0, 0
    while offset + 8 <= size.value:
        relationship = wintypes.DWORD.from_buffer(buffer, offset).value
        record_size = wintypes.DWORD.from_buffer(buffer, offset + 4).value
        if record_size < 8 or offset + record_size > size.value:
            return None
        count += relationship == 0
        offset += record_size
    return count if offset == size.value and count else None


def configure_windows_cpu_detection():
    if sys.platform != 'win32':
        return
    from joblib.externals.loky.backend import context
    # Older bundled loky versions call WMIC, removed from recent Windows.
    # Populate their existing cache with the native count. Leave user worker
    # limits and affinity handling in cpu_count() intact.
    if hasattr(context, 'physical_cores_cache') and context.physical_cores_cache is None:
        count = windows_physical_cpu_count()
        if count is not None:
            context.physical_cores_cache = count
