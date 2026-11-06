"""Modern window shell for the shared standalone snapper workflow."""
from ui.pixel_snapper import SnapperContents
from .modal import Dialog
from . import theme as T


class ModernPixelSnapperDialog(SnapperContents, Dialog):
    def __init__(self, parent, source, config=None):
        super().__init__(parent, 'Pixel Snapper', icon='grid_fine', size=(1100, 650), resizable=True)
        self.minsize(900, 560)
        self._initialize_snapper(source, config)
        for button in (self.preview_button, self.apply_button):
            button.configure(fg_color=T.accent(), hover_color=T.accent_hover(), text_color=T.accent_on())
        self.cancel_button.configure(fg_color=T.c('card_alt'), hover_color=T.c('card_hi'), text_color=T.c('text'))
        self.algorithm_menu.configure(fg_color=T.c('card_alt'), button_color=T.c('card_hi'),
                                      button_hover_color=T.c('border'), text_color=T.c('text'))
        self.auto_check.configure(fg_color=T.accent(), hover_color=T.accent_hover(), checkmark_color=T.accent_on())
        for button in (self.reset_palette_button, self.fit_button, self.add_color_button, self.delete_color_button):
            button.configure(fg_color=T.c('card_alt'), hover_color=T.c('card_hi'), text_color=T.c('text'))
