# Production GUI migration

The new default uses the prototype's presentation, not its simulated actions.
The original GUI remains available through `--old-design` / `--old_design`.
Production must never import `_gui_preview`.

## Behavior inventory

| Area | Production behavior to preserve | Migration decision |
| --- | --- | --- |
| Source | Image decoding, real transparency detection, recent paths, video first/random frames | Source panel; real files, no generated scene or invented video metadata |
| Pixelization | Shared nearest-neighbor implementation, even dimensions based on the **shorter** edge, optional neural model | Label the real size semantics; do not copy mockup's longest-edge resize |
| Neural | Lazy model loading; disallow transparent inputs | Preserve capability gating and surface errors |
| Editor setup | Target grid, source pan/zoom, fit-to-grid, alpha cutoff, conversion | Use `ui.pixelization_editor.PixelizationEditorDialog`, never the prototype editor |
| Editor tools | Brush strokes/interpolation, connected wand replacement, color picker, Alt picker, right-click/Shift pan, grid, undo/redo | Retain production canvas and shortcuts |
| Editor result | Missing cells become transparent RGBA; apply/cancel; geometry and tool preferences | Preserve result alpha, invalidate previous dithering and cache |
| Palette | Generated median-cut/k-means/uniform, persistent custom palettes, Lospec import, extraction from another image | Real palette manager and existing creation/import workflows |
| Dithering | Actual enum modes and parameter schemas, gamma, source alpha, latest-request preview | Use `ImageDitherer` and `ui.preview_jobs`, never mock mode definitions |
| Viewer | Original/pixelized/dithered, zoom, pan, fit | Prototype canvas presentation with explicit state selection and comparison |
| Image export | Best processed result, integer nearest-neighbor upscale, filenames, remembered directory, alpha-safe formats | Real save workflow, no fake clipboard/export success |
| Video export | Selected palette/gamma/parameters, optional pixelization/upscale, bounded streaming, progress, original audio/subtitles | Real `VideoProcessor`; capture all Tk settings before worker execution |
| Preferences | Existing settings survive; new installations start light | Persist appearance and design system; retain old GUI configuration |
| Window controls | Custom/native caption option, DPI-safe bounds, modal-safe theme changes, orderly shutdown | Promote tested chrome helpers; validate real Windows widgets |

## Prototype controls that must not ship as fake features

The mock video scrubber, made-up frame count, codec/CRF controls without backend
support, fake clipboard action, synthetic gallery results, cache preference
without a cache, and simulated processing/export are not production features.
Only expose controls backed by an actual operation. Mode parameter ranges come
from the library, not `demo.MODE_PARAMS`.

## Validation checkpoints

1. Importable production UI components and launch routing; CLI unchanged.
2. Real image loading, pixelization, palettes, dithering and export, including alpha.
3. Full production editor apply/cancel/history behavior; no mock editor imports.
4. Real video processing settings and progress; no worker-thread Tk access.
5. Light first-run theme, persisted preferences, modal theme/reopen/close checks.
6. Existing regression suite plus new GUI integration checks; switch default only
   once these pass. Keep the legacy launch path as a supported fallback.
