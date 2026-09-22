"""Compatibility exports for GUI components. Implementations live in ui/."""

from ui.image_view import (
    ZoomableImage,
    ImageComparisonView,
)
from ui.pixelization_editor import (
    PixelizationEditorCanvas,
    PixelizationEditorDialog,
)
from ui.palettes import (
    PalettePreview,
    ColorPickerGrid,
    HSVColorPickerDialog,
    CustomPaletteCreator,
    PaletteImagePreviewDialog,
)
from ui.status import (
    ProgressDialog,
    StatusBar,
)
from ui.dither_settings import (
    DitherSettingsDialog,
)

__all__ = ['ZoomableImage', 'PixelizationEditorCanvas', 'PalettePreview', 'ProgressDialog', 'ColorPickerGrid', 'StatusBar', 'ImageComparisonView', 'HSVColorPickerDialog', 'CustomPaletteCreator', 'PaletteImagePreviewDialog', 'DitherSettingsDialog', 'PixelizationEditorDialog']
