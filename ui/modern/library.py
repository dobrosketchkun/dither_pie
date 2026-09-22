"""Bundled palette swatches from the design prototype."""
def _hexlist(s: str):
    out = []
    for part in s.split():
        part = part.lstrip("#")
        out.append(tuple(int(part[i:i + 2], 16) for i in (0, 2, 4)))
    return out

LIBRARY_PALETTES = [('Game Boy DMG', 'lospec', _hexlist('0f380f 306230 8bac0f 9bbc0f')), ('PICO-8', 'lospec', _hexlist('000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa')), ('Sweetie 16', 'lospec', _hexlist('1a1c2c 5d275d b13e53 ef7d57 ffcd75 a7f070 38b764 257179 29366f 3b5dc9 41a6f6 73eff7 f4f4f4 94b0c2 566c86 333c57')), ('CGA 16', 'lospec', _hexlist('000000 0000aa 00aa00 00aaaa aa0000 aa00aa aa5500 aaaaaa 555555 5555ff 55ff55 55ffff ff5555 ff55ff ffff55 ffffff')), ('Nord', 'custom', _hexlist('2e3440 3b4252 434c5e 4c566a d8dee9 e5e9f0 eceff4 8fbcbb 88c0d0 81a1c1 5e81ac bf616a d08770 ebcb8b a3be8c b48ead')), ('Dracula', 'custom', _hexlist('282a36 44475a f8f8f2 6272a4 8be9fd 50fa7b ffb86c ff79c6 bd93f9 ff5555 f1fa8c')), ('Solarized', 'custom', _hexlist('002b36 073642 586e75 657b83 839496 93a1a1 eee8d5 fdf6e3 b58900 cb4b16 dc322f d33682 6c71c4 268bd2 2aa198 859900')), ('1-Bit Ink', 'custom', _hexlist('1b1b1b f2f0e6'))]
