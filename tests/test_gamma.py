import unittest

import numpy as np
from PIL import Image

from dithering_lib import ColorReducer, DitherMode, ImageDitherer


class GammaTests(unittest.TestCase):
    def test_every_gray_palette_value_round_trips_exactly(self):
        ramp = np.arange(256, dtype=np.uint8)
        source = Image.fromarray(np.repeat(ramp[None, :, None], 3, axis=2))
        palette = [(value, value, value) for value in range(256)]
        result = ImageDitherer(palette=palette, dither_mode=DitherMode.NONE,
                               use_gamma=True).apply_dithering(source)
        np.testing.assert_array_equal(np.array(result), np.array(source))

    def test_generated_palette_is_srgb_and_matches_explicit_palette(self):
        pixels = np.random.RandomState(12).randint(0, 256, (8, 8, 3)).astype(np.uint8)
        source = Image.fromarray(pixels)
        palette = ColorReducer.reduce_colors(source, 4)
        generated = ImageDitherer(num_colors=4, use_gamma=True)
        result = generated.apply_dithering(source)
        explicit = ImageDitherer(palette=palette, use_gamma=True).apply_dithering(source)
        self.assertEqual(generated.palette, palette)
        np.testing.assert_array_equal(np.array(result), np.array(explicit))

    def test_all_modes_preserve_dark_palette_colors_and_alpha(self):
        palette = [(1, 7, 13), (20, 30, 40), (90, 140, 190), (210, 220, 230)]
        pixels = np.random.RandomState(9).randint(0, 256, (8, 8, 4)).astype(np.uint8)
        pixels[0, :, 3] = 0
        for mode in DitherMode:
            with self.subTest(mode=mode.value):
                params = {'size': 4} if mode == DitherMode.BLUE_NOISE else {}
                result = ImageDitherer(palette=palette, use_gamma=True,
                                       dither_mode=mode, dither_params=params).apply_dithering(Image.fromarray(pixels))
                output = np.array(result)
                np.testing.assert_array_equal(output[..., 3], pixels[..., 3])
                self.assertTrue(set(map(tuple, output[..., :3].reshape(-1, 3))).issubset(set(palette)))

    def test_one_color_palette_works_for_all_modes(self):
        source = Image.new('RGB', (4, 4), (100, 100, 100))
        for mode in DitherMode:
            with self.subTest(mode=mode.value):
                params = {'size': 4} if mode == DitherMode.BLUE_NOISE else {}
                result = ImageDitherer(palette=[(1, 5, 9)], use_gamma=True,
                                       dither_mode=mode, dither_params=params).apply_dithering(source)
                np.testing.assert_array_equal(np.array(result), np.full((4, 4, 3), (1, 5, 9)))
