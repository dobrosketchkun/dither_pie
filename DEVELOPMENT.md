# Development

Run commands from the repository root. The application currently resolves palette
files and neural weights relative to the working directory.

## Environment and checks

The refactoring baseline was verified with Python 3.10 in `general_env_1`:

```powershell
conda activate general_env_1
python -m unittest discover -s tests -v
python -X utf8 dither_pie.py --help
python dither_pie.py --example-config
```

For a new environment, install `requirements.txt`. For neural pixelization,
install `requirements-neural.txt` and obtain the model weights described in the
README. These files list dependencies; they are not a version lock. FFmpeg and
ffprobe are external executables required for video processing. Numba is optional;
the dithering library falls back to its Python implementations when unavailable.

The tests generate small images in temporary directories. They cover all dithering
modes, alpha preservation, palette masking, CLI image processing and validation,
frame workers, command-line entry points, and GUI helpers. They do not open GUI
windows, download weights, or process the user's media.

Interactive dialog behavior and full neural/video rendering still require separate
checks. Passing this suite is a regression baseline, not a claim that all existing
bugs have been fixed.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `dither_pie.py` | Select CLI or GUI entry point |
| `dither_cli.py` | Arguments, logging, batch orchestration, and CLI output |
| `processing_config.py` | Load and validate processing jobs; detect input mode |
| `config_manager.py` | Persistent GUI preferences and recent files |
| `dithering_lib.py` | Dithering strategies, palette reduction, gamma, and alpha handling |
| `pixelization.py` | Shared regular/neural pixelization used by images and video |
| `video_processor.py` | Frame processing, multiprocessing, and FFmpeg orchestration |
| `dither_pie_gui.py` | Main window, application state, and workflow coordination |
| `ui/image_view.py` | Image display, zoom, pan, and comparison |
| `ui/pixelization_editor.py` | Pixel editor canvas and dialog |
| `ui/palettes.py` | Palette widgets and palette creation dialogs |
| `ui/dither_settings.py` | Algorithm parameter dialog |
| `ui/status.py` | Status bar and progress dialog |
| `utils.py` | Palette storage/import and file-format helpers |

`gui_components.py` re-exports the original public GUI classes. Configuration
functions remain importable from `dither_cli`, and pixelization helpers remain
importable from `video_processor`. Prefer the dedicated modules in new code.

Processing-job configuration and GUI preferences are separate concerns: the former
describes a requested operation, while the latter stores interface state and defaults.
Do not make processing validation depend on a running window or CLI logger.

## Follow-up reliability work

Keep behavior changes separate from structural moves and add a regression case for
each fix. The initial review identified these follow-ups:

- Correct palette handling and precision in gamma mode.
- Queue the latest preview request and reject stale results when settings change.
- Normalize validated numbers and reject unsupported size boundaries.
- Return failure for partially failed or interrupted batches.
- Parse named ffprobe fields rather than guessing duration/frame count by magnitude.
- Reuse video workers and replace whole-video PNG extraction with bounded streaming.
- Make redirected CLI help safe on Windows legacy encodings. Its existing Unicode
  banner fails under CP1252; the entry-point smoke tests explicitly use UTF-8 mode.

The structural refactor intentionally preserves these existing behaviors so its
commits can be reviewed independently of the fixes.
