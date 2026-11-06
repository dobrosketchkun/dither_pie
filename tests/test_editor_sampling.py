import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image
from ui.pixelization_editor import sample_editor_grid, PixelizationEditorCanvas
from ui.image_view import render_visible_region, ZoomableImage


class EditorSamplingTests(unittest.TestCase):
    @staticmethod
    def zoom_canvas(zoom=1, grid=1, alt=True, mode='preview'):
        canvas = SimpleNamespace(original_image=Image.new('RGB', (128, 96)),
            zoom_factor=zoom, preview_grid_scale=grid, alt_zoom_active=alt, mode=mode,
            offset_x=17.25, offset_y=-23.5, preview_grid_offset_x=-11, preview_grid_offset_y=29,
            grid_w=64, grid_h=48, winfo_width=lambda: 801, winfo_height=lambda: 601,
            update_view=lambda: None)
        canvas._get_image_draw_rect = lambda: PixelizationEditorCanvas._get_image_draw_rect(canvas)
        canvas._get_preview_grid_rect = lambda: PixelizationEditorCanvas._get_preview_grid_rect(canvas)
        return canvas

    def test_zoom_keeps_cursor_point_fixed_in_image_and_coupled_grid(self):
        for alt, mode in ((False, 'preview'), (True, 'preview'), (True, 'edit')):
            for delta in (120, -120):
                for state in (0, 1):
                    with self.subTest(alt=alt, mode=mode, delta=delta, state=state):
                        canvas = self.zoom_canvas(zoom=1.7, grid=1.3, alt=alt, mode=mode)
                        event = SimpleNamespace(x=173, y=219, num=0, delta=delta, state=state)
                        image = canvas._get_image_draw_rect()
                        grid = canvas._get_preview_grid_rect()
                        image_point = ((event.x - image[0]) / image[2], (event.y - image[1]) / image[3])
                        grid_point = ((event.x - grid[0]) / grid[2], (event.y - grid[1]) / grid[3])
                        PixelizationEditorCanvas.zoom(canvas, event)
                        image_after = canvas._get_image_draw_rect()
                        self.assertAlmostEqual(image_after[0] + image_point[0] * image_after[2], event.x)
                        self.assertAlmostEqual(image_after[1] + image_point[1] * image_after[3], event.y)
                        grid_after = canvas._get_preview_grid_rect()
                        if alt and mode == 'preview':
                            self.assertAlmostEqual(grid_after[0] + grid_point[0] * grid_after[2], event.x)
                            self.assertAlmostEqual(grid_after[1] + grid_point[1] * grid_after[3], event.y)
                        else:
                            self.assertEqual(grid, grid_after)

    def test_shift_pan_uses_same_increment_for_grid_and_image(self):
        canvas = SimpleNamespace(mode='preview', offset_x=-12000, offset_y=350,
            preview_grid_offset_x=42, preview_grid_offset_y=-18, _last_pan_event=(200, 150),
            pan_start_x=-9999, pan_start_y=-9999, update_view=Mock(),
            _shift_down=PixelizationEditorCanvas._shift_down)
        PixelizationEditorCanvas.pan(canvas, SimpleNamespace(x=225, y=140, state=1))
        self.assertEqual((canvas.offset_x, canvas.offset_y), (-11975, 340))
        self.assertEqual((canvas.preview_grid_offset_x, canvas.preview_grid_offset_y), (67, -28))
        PixelizationEditorCanvas.pan(canvas, SimpleNamespace(x=230, y=143, state=0))
        self.assertEqual((canvas.offset_x, canvas.offset_y), (-11970, 343))
        self.assertEqual((canvas.preview_grid_offset_x, canvas.preview_grid_offset_y), (67, -28))
        canvas._last_pan_event = None
        canvas.start_pan = Mock()
        PixelizationEditorCanvas.pan(canvas, SimpleNamespace(x=900, y=600, state=1))
        canvas.start_pan.assert_called_once()
        self.assertEqual((canvas.offset_x, canvas.offset_y), (-11970, 343))

    def test_shift_press_overrides_picker_in_preview(self):
        canvas = SimpleNamespace(mode='preview', tool='picker', start_pan=Mock(),
            _pick_color_at_canvas=Mock(), update_view=Mock(),
            _shift_down=PixelizationEditorCanvas._shift_down)
        event = SimpleNamespace(x=200, y=150, state=1)
        PixelizationEditorCanvas._on_left_down(canvas, event)
        canvas.start_pan.assert_called_once_with(event)
        canvas._pick_color_at_canvas.assert_not_called()

    def test_visible_render_matches_full_nearest_resize_and_transparency(self):
        source = Image.fromarray(np.random.default_rng(4).integers(0, 256, (9, 12, 4), dtype=np.uint8))
        for zoom in (.5, 1, 1.7, 3, 30):
            for offsets in ((0, 0), (3, -2), (.25, -.5)):
                with self.subTest(zoom=zoom, offsets=offsets):
                    tile, _, _, (left, top) = render_visible_region(source, zoom, (13, 11), offsets)
                    full = source.resize((int(source.width * zoom), int(source.height * zoom)), Image.Resampling.NEAREST)
                    expected = full.crop((left, top, left + tile.width, top + tile.height))
                    np.testing.assert_array_equal(np.array(tile), np.array(expected))
                    expected = ZoomableImage._composite_for_display(full).crop(
                        (left, top, left + tile.width, top + tile.height))
                    actual = ZoomableImage._composite_for_display(tile, (left, top))
                    np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_deep_zoom_never_allocates_the_full_enlarged_image(self):
        image = Image.new('RGB', (2048, 2048))
        source = SimpleNamespace(width=image.width, height=image.height, resize=Mock(wraps=image.resize))
        tile, _, _, _ = render_visible_region(source, 30, (800, 600), (.25, -.5))
        self.assertLessEqual(tile.width, 801)
        self.assertLessEqual(tile.height, 601)
        self.assertEqual(source.resize.call_args.args[0], tile.size)
        self.assertIsNone(render_visible_region(source, 30, (800, 600), (100000, 0)))
        self.assertEqual(source.resize.call_count, 1)

    def test_only_visible_grid_lines_are_created(self):
        lines = []
        canvas = SimpleNamespace(grid_w=2048, grid_h=2048, winfo_width=lambda: 800,
            winfo_height=lambda: 600, create_line=lambda *args, **kwargs: lines.append(args))
        PixelizationEditorCanvas._draw_visible_grid(canvas, -30000, -30000, 30)
        self.assertLessEqual(len(lines), 800 // 30 + 600 // 30 + 2)
        for x0, y0, x1, y1 in lines:
            self.assertTrue(0 <= x0 <= 800 and 0 <= x1 <= 800)
            self.assertTrue(0 <= y0 <= 600 and 0 <= y1 <= 600)

    def test_hover_only_redraws_highlight(self):
        canvas = SimpleNamespace(mode='edit', highlight_cell=(0, 0),
            _canvas_to_cell=lambda x, y: (1, 1), delete=Mock(), _draw_highlight=Mock(), update_view=Mock())
        PixelizationEditorCanvas._on_motion(canvas, SimpleNamespace(x=10, y=10))
        canvas.update_view.assert_not_called()
        canvas.delete.assert_called_once_with('highlight')
        canvas._draw_highlight.assert_called_once()

    def test_alt_zoom_stops_image_and_grid_together_at_either_limit(self):
        for zoom, grid, direction, expected_zoom, expected_grid in (
                (29, 1, 120, 30, 30 / 29),
                (30, 1, 120, 30, 1),
                (1, 9.8, 120, 10 / 9.8, 10),
                (1, 10, 120, 1, 10),
                (.011, 1, -120, .01, .01 / .011),
                (.01, 1, -120, .01, 1),
                (1, .105, -120, .1 / .105, .1),
                (1, .1, -120, 1, .1)):
            for state in (0, 1):  # Normal and Shift fine increments.
                with self.subTest(zoom=zoom, grid=grid, direction=direction, state=state):
                    canvas = self.zoom_canvas(zoom=zoom, grid=grid)
                    event = SimpleNamespace(num=0, delta=direction, state=state, x=173, y=219)
                    ratio = zoom / grid
                    for _ in range(100):
                        PixelizationEditorCanvas.zoom(canvas, event)
                        self.assertAlmostEqual(canvas.zoom_factor / canvas.preview_grid_scale, ratio)
                    self.assertAlmostEqual(canvas.zoom_factor, expected_zoom)
                    self.assertAlmostEqual(canvas.preview_grid_scale, expected_grid)

    def test_regular_zoom_and_edit_mode_do_not_change_preview_grid(self):
        for alt, mode in ((False, 'preview'), (True, 'edit')):
            canvas = self.zoom_canvas(zoom=29, alt=alt, mode=mode)
            PixelizationEditorCanvas.zoom(canvas, SimpleNamespace(num=0, delta=120, state=0, x=173, y=219))
            self.assertEqual(canvas.zoom_factor, 30)
            self.assertEqual(canvas.preview_grid_scale, 1)

    def test_grid_uses_captured_transform_and_alpha_cutoff(self):
        source = Image.new('RGBA', (4, 4), (20, 40, 60, 255))
        source.putpixel((1, 1), (255, 0, 0, 128))
        source.putpixel((3, 1), (1, 2, 3, 129))
        grid = sample_editor_grid(source, 2, 2, (0, 0, 1), (0, 0, 4, 4, 2), 128)
        self.assertEqual(grid, [[None, (1, 2, 3)], [(20, 40, 60), (20, 40, 60)]])

    def test_grid_cells_outside_source_remain_absent(self):
        source = Image.new('RGBA', (2, 2), (20, 40, 60, 255))
        grid = sample_editor_grid(source, 2, 2, (0, 0, 1), (0, 0, 4, 4, 2), 0)
        self.assertEqual(grid, [[(20, 40, 60), None], [None, None]])
