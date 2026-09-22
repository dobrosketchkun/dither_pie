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
frame workers, command-line entry points, and GUI helpers. Preview workflow tests
use simulated widgets and slow background jobs; they check Apply/Cancel and
settings changes without opening windows. When FFmpeg and ffprobe are available,
the video tests generate tiny clips to check streaming, audio/subtitles, rotation,
variable and fractional rates, interruption, and cleanup. They do not download
weights or process the user's media.

Visual GUI behavior and actual neural inference still require separate checks.
The neural video test uses a substitute model to verify model reuse and scheduling;
it does not validate model weights or GPU behavior.

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
| `ui/preview_jobs.py` | Latest-request preview scheduling, independent of Tk |
| `utils.py` | Palette storage/import and file-format helpers |

`gui_components.py` re-exports the original public GUI classes. Configuration
functions remain importable from `dither_cli`, and pixelization helpers remain
importable from `video_processor`. Prefer the dedicated modules in new code.

Processing-job configuration and GUI preferences are separate concerns: the former
describes a requested operation, while the latter stores interface state and defaults.
Do not make processing validation depend on a running window or CLI logger.

## Reliability behavior

- Palettes remain in sRGB. Gamma processing uses floating-point linear values until
  final encoding, preserving the selected palette's exact RGB values and alpha.
- Previews allow one active job and one replaceable pending request. Workers never
  access Tk; the UI thread polls results and rejects stale work. Apply waits for
  the selected result, and closing the dialog invalidates outstanding work.
- Job validation normalizes integer strings, rejects fractions/booleans as integer
  settings, and requires pixelization sizes of at least 2. Validation leaves the
  caller's configuration unchanged.
- Batches return success only when all files succeed. Interruption stops the batch
  and propagates to the CLI's exit code 130; partial failure exits with code 1.
- Video metadata comes from named ffprobe JSON fields, including exact rational
  frame rates and displayed dimensions after rotation. Invalid essential metadata
  produces an error instead of invented values.
- Video frames flow through FFmpeg pipes in bounded batches. One CPU worker pool
  is reused per video; neural inference stays in the calling process with one model.
  The batch limit targets at most 64 MiB of input/output pixel payloads (or one frame
  if larger); process overhead and algorithm workspaces are additional.
- Video input with variable frame timing is resampled to its average frame rate.
  Constant fractional rates are preserved exactly. Original variable timestamps
  are not retained. Audio/subtitles are copied when the destination container
  supports their codecs, trimmed to the rendered video duration.
- Only compressed intermediate video and mux output are staged on disk. The output
  is replaced after success. Failed frames fail the operation instead of silently
  substituting frozen frames. Errors and interruption reap workers/FFmpeg and remove
  staging files, preserving an existing destination.

Keep future behavior changes separate from structural moves and add a regression
case for each fix.

## Remaining checks

Redirected CLI help still needs a Windows legacy-encoding fix: its existing Unicode
banner fails under CP1252, so entry-point smoke tests explicitly use UTF-8 mode.
Interactive visual QA and real neural/GPU inference are outside the automated suite.
