# CLI Implementation Plan for Dither Pie

**Goal:** Add CLI mode to dither_pie.py while keeping GUI functionality intact.

**Strategy:** 
- Move GUI code to `dither_pie_gui.py`
- Keep `dither_pie.py` as entry point (GUI if no args, CLI if args)
- Use JSON configs for CLI
- Use `logging` + `rich` for all CLI output
- Keep neural model loaded during batch operations

---

## ⚠️ IMPORTANT: NO TEST FILES

**DO NOT CREATE SEPARATE TEST `.py` FILES FOR TESTING!**

When the plan says "Test: ..." → Ask the user to test it manually.

Examples:
- ❌ Don't create `test_import.py` to test imports
- ❌ Don't create `test_config.py` to test config validation
- ❌ Don't create `test_cli.py` for any testing

✅ Instead: Make the change, then ask user to run the command and report results.

---

## Phase 1: Code Reorganization

### 1.1 Create GUI Module
- [ ] Create new file `dither_pie_gui.py`
- [ ] Move `DitheringApp` class from `dither_pie.py` to `dither_pie_gui.py`
- [ ] Verify all imports are present in `dither_pie_gui.py`
- [ ] Test: Import `DitheringApp` from new location works

### 1.2 Create Entry Point
- [ ] Modify `dither_pie.py` to become entry point only
- [ ] Add argument detection logic:
  ```python
  if len(sys.argv) == 1:
      # Launch GUI
  else:
      # Launch CLI
  ```
- [ ] Test: Running `python dither_pie.py` launches GUI
- [ ] Test: Running `python dither_pie.py --help` shows CLI help (stub for now)

### 1.3 Create CLI Module Structure
- [ ] Create new file `dither_cli.py`
- [ ] Add basic structure:
  - [ ] Imports (including `logging`, `rich`)
  - [ ] `setup_logging()` function
  - [ ] `main(args)` function stub
- [ ] Test: Can import from `dither_cli` without errors

---

## Phase 2: Logging Infrastructure

### 2.1 Setup Rich Logging
- [ ] Install dependencies: `pip install rich` (if not present)
- [ ] Create `setup_logging()` in `dither_cli.py`:
  - [ ] Configure logging level
  - [ ] Setup RichHandler
  - [ ] Format string configuration
- [ ] Create logger instance: `logger = logging.getLogger('dither_pie')`
- [ ] Test: Logger outputs colored messages

### 2.2 Progress Tracking
- [ ] Create `CLIProgressCallback` class in `dither_cli.py`
  - [ ] Uses `rich.progress.Progress`
  - [ ] Compatible with `VideoProcessor.progress_callback` signature
  - [ ] Methods: `__init__`, `update(fraction, message)`, `finish()`
- [ ] Test: Progress bar displays and updates correctly

---

## Phase 3: JSON Config System

### 3.1 Define Config Schema
- [ ] Create `config_schema.py` (or section in `dither_cli.py`)
- [ ] Define schema structure:
  ```python
  SCHEMA = {
      "input": str,          # Required
      "output": str,         # Required
      "mode": str,           # "image", "video", "folder"
      "pixelization": {...},
      "dithering": {...},
      "palette": {...},
      ...
  }
  ```
- [ ] Document all fields with comments

### 3.2 Config Validation
- [ ] Create `validate_config(config_dict)` function
  - [ ] Check required fields present
  - [ ] Validate field types
  - [ ] Validate enum values (dither modes, palette sources)
  - [ ] Check file paths exist (for input)
  - [ ] Validate numeric ranges (colors, max_size, etc.)
  - [ ] Return clear error messages with field names
- [ ] Create `load_config(json_path)` function
  - [ ] Load JSON file
  - [ ] Call `validate_config()`
  - [ ] Return validated config or raise error
- [ ] Test: Valid config loads successfully
- [ ] Test: Invalid configs fail with clear messages

### 3.3 Default Config Template
- [ ] Create `example_config.json` with all options documented
- [ ] Add comments explaining each field (as JSON doesn't support comments, use a companion `.md` file)
- [ ] Test: Example config validates and runs

---

## Phase 4: Core CLI Functions

### 4.1 Palette Setup Helper
- [ ] Create `setup_palette_from_config(palette_config, source_image)` in `dither_cli.py`
  - [ ] Handle "median_cut" source
  - [ ] Handle "kmeans" source
  - [ ] Handle "uniform" source
  - [ ] Handle custom palette by name (from palette.json)
  - [ ] Handle explicit colors list (hex)
  - [ ] Handle "file:path.png" (extract from image)
  - [ ] Use `ColorReducer` and `PaletteManager`
  - [ ] Return `List[Tuple[int, int, int]]`
- [ ] Test each palette source type
- [ ] Log palette source and color count

### 4.2 Single Image Processing
- [ ] Create `process_single_image(config)` in `dither_cli.py`
  - [ ] Load image with PIL
  - [ ] Log image dimensions
  - [ ] Apply pixelization (if configured):
    - [ ] "none" → skip
    - [ ] "regular" → call `pixelize_regular()`
    - [ ] "neural" → use `NeuralPixelizer` instance
  - [ ] Log pixelization result
  - [ ] Setup palette using helper
  - [ ] Create `ImageDitherer` with config params
  - [ ] Apply dithering
  - [ ] Log dithering complete
  - [ ] Apply final resize (if configured)
  - [ ] Save output
  - [ ] Log output path and file size
- [ ] Test with various config combinations
- [ ] Handle errors gracefully with logging

### 4.3 Single Video Processing
- [ ] Create `process_single_video(config, neural_pixelizer=None)` in `dither_cli.py`
  - [ ] Accept optional pre-loaded `NeuralPixelizer` instance
  - [ ] Create `CLIProgressCallback` instance
  - [ ] Create `VideoProcessor` with callback
  - [ ] Setup palette (from first frame or config)
  - [ ] Create `ImageDitherer`
  - [ ] Prepare pixelization tuple for VideoProcessor:
    - [ ] `None` if no pixelization
    - [ ] `("regular", max_size)` for regular
    - [ ] `("neural", max_size)` for neural
  - [ ] Call `process_video_streaming()`
  - [ ] Log completion and output file info
- [ ] Test with short video
- [ ] Test with/without pixelization

---

## Phase 5: Batch Processing

### 5.1 Folder Processing (Images)
- [ ] Create `process_folder_images(config, pattern='*.png')` in `dither_cli.py`
  - [ ] Glob files from input folder
  - [ ] Log total file count
  - [ ] Optionally pre-load `NeuralPixelizer` if needed (check config)
  - [ ] Loop through files:
    - [ ] Log current file (e.g., "[3/10] Processing image.png")
    - [ ] Create per-file config (update input/output paths)
    - [ ] Call `process_single_image()`
    - [ ] Handle errors per-file (continue on error)
  - [ ] Log summary (total processed, failed, time)
- [ ] Test with folder of images
- [ ] Test error handling (one bad file shouldn't stop batch)

### 5.2 Folder Processing (Videos)
- [ ] Create `process_folder_videos(config, pattern='*.mp4')` in `dither_cli.py`
  - [ ] Glob video files
  - [ ] Pre-load `NeuralPixelizer` once if needed
  - [ ] Loop through videos:
    - [ ] Pass shared pixelizer instance to `process_single_video()`
  - [ ] Log summary
- [ ] Test with multiple videos

### 5.3 Auto Mode Detection
- [ ] Create `detect_mode(config)` function
  - [ ] If `config['mode']` explicitly set, use it
  - [ ] Otherwise, auto-detect:
    - [ ] File + video extension → "video"
    - [ ] File + image extension → "image"
    - [ ] Directory → "folder"
  - [ ] Raise error if can't detect
- [ ] Update `main()` to use detection
- [ ] Test auto-detection works

---

## Phase 6: Integration & Testing

### 6.1 Wire Up Entry Point
- [ ] In `dither_pie.py`, implement CLI argument parsing:
  - [ ] Positional arg: config file path
  - [ ] Optional flags: `--verbose`, `--quiet`, `--log-file`
  - [ ] `--help` shows usage
  - [ ] `--example-config` outputs example JSON
- [ ] Call `dither_cli.main(args)`
- [ ] Test: `python dither_pie.py config.json` runs CLI
- [ ] Test: `python dither_pie.py` runs GUI

### 6.2 End-to-End Tests
- [ ] Test: Single image, no pixelization
- [ ] Test: Single image, regular pixelization
- [ ] Test: Single image, neural pixelization
- [ ] Test: Single video, regular pixelization
- [ ] Test: Single video, neural pixelization
- [ ] Test: Folder of images (mixed success/fail)
- [ ] Test: All dither modes work
- [ ] Test: All palette sources work
- [ ] Test: Parameter configs work (error_diffusion variant, bayer size, halftone params)
- [ ] Test: Invalid config shows clear error
- [ ] Test: Missing input file shows clear error
- [ ] Test: Output directory created if missing

### 6.3 Error Handling Review
- [ ] File not found errors are caught and logged
- [ ] Invalid JSON shows parse error with line number
- [ ] Invalid config fields show which field is wrong
- [ ] Processing errors don't crash, just log and continue (in batch mode)
- [ ] Video processing errors are handled gracefully
- [ ] Neural model loading errors are clear

---

## Phase 7: Documentation & Polish

### 7.1 Example Configs
- [ ] Create `examples/` directory
- [ ] `examples/image_basic.json` - simple image dithering
- [ ] `examples/image_neural.json` - with neural pixelization
- [ ] `examples/video_batch.json` - video processing
- [ ] `examples/folder_batch.json` - folder of images
- [ ] `examples/advanced.json` - all options demonstrated
- [ ] Each with companion `.md` explaining options

### 7.2 CLI Help
- [ ] Add `--help` output with:
  - [ ] Usage examples
  - [ ] Config file format overview
  - [ ] List of available dither modes
  - [ ] List of palette sources
  - [ ] Link to example configs
- [ ] Add `--list-modes` to show all dither modes with descriptions
- [ ] Add `--list-palettes` to show available custom palettes

### 7.3 Update README
- [ ] Add CLI section to `README_NEW.md`
- [ ] Document JSON config schema
- [ ] Provide usage examples
- [ ] Document batch processing workflow

---

## Phase 8: Optimization & Advanced Features

### 8.1 Performance
- [ ] Verify neural model is loaded once for folder batches
- [ ] Verify blue noise cache works in CLI
- [ ] Add optional parallel processing for image folders (multiprocessing)

### 8.2 Advanced Options
- [ ] Add `--dry-run` flag (validate config, show what would be processed)
- [ ] Add `--overwrite` flag (skip existing output files by default)
- [ ] Add resume capability for interrupted folder processing

### 8.3 Config Inheritance
- [ ] Support "base" config + override patterns:
  ```json
  {
    "base": "base_config.json",
    "input": "override_this.png"
  }
  ```
- [ ] Useful for batch jobs with slight variations

---

## Testing Checklist

### Quick Smoke Tests (run after each phase)
- [ ] GUI still launches: `python dither_pie.py`
- [ ] CLI help works: `python dither_pie.py --help`
- [ ] Example config runs: `python dither_pie.py examples/image_basic.json`

### Full Test Suite (before declaring done)
- [ ] All 12 dither modes work in CLI
- [ ] All palette generation methods work
- [ ] All pixelization modes work
- [ ] Folder batch processing completes
- [ ] Video processing completes
- [ ] Error cases show helpful messages
- [ ] Logging levels work (verbose/quiet)
- [ ] GUI unaffected by CLI changes

---

## Files to Create/Modify

### New Files
- [ ] `dither_pie_gui.py` - GUI code moved here
- [ ] `dither_cli.py` - CLI implementation
- [ ] `config_schema.py` (optional) - schema definitions
- [ ] `example_config.json` - example configuration
- [ ] `examples/` directory with sample configs
- [ ] `CLI_USAGE.md` - CLI documentation

### Modified Files
- [ ] `dither_pie.py` - becomes entry point router
- [ ] `README_NEW.md` - add CLI section

### Untouched Files (no changes needed)
- [ ] `dithering_lib.py` - already CLI-ready
- [ ] `video_processor.py` - already decoupled
- [ ] `utils.py` - already decoupled
- [ ] `config_manager.py` - already CLI-ready
- [ ] `gui_components.py` - GUI only, no CLI needs it

---

## Dependencies to Add

- [ ] `rich` - for beautiful terminal output and progress bars
- [ ] Verify no new deps needed (everything else should be installed)

---

## Risk Mitigation

### Before Starting
- [ ] Commit current working state to git
- [ ] Create feature branch: `git checkout -b feature/cli-mode`

### During Development
- [ ] Test GUI after each major change
- [ ] Keep checkpoints (git commits) after each phase
- [ ] If something breaks, can revert to last checkpoint

### Final Review
- [ ] Run full test suite
- [ ] Verify no GUI regressions
- [ ] Merge to main only when all checks pass

---

## Success Criteria

✅ CLI mode works for single images, videos, and folders  
✅ GUI mode still works exactly as before  
✅ No code duplication (core logic reused)  
✅ JSON configs are validated with clear errors  
✅ Rich logging provides excellent feedback  
✅ Neural model loaded once for batch operations  
✅ All dithering modes and features accessible from CLI  
✅ Documentation complete and examples provided  

---

## Estimated Time per Phase

- Phase 1: 30-45 min
- Phase 2: 30 min
- Phase 3: 45-60 min
- Phase 4: 90-120 min
- Phase 5: 60-90 min
- Phase 6: 45-60 min
- Phase 7: 30-45 min
- Phase 8: 60 min (optional)

**Total: 6-8 hours** (without Phase 8)

---

## Notes & Gotchas

1. **Import Cycles**: Watch for circular imports between `dither_pie.py`, `dither_pie_gui.py`, and `dither_cli.py`. Keep entry point (`dither_pie.py`) as thin as possible.

2. **VideoProcessor Callback**: Current signature is `callback(fraction, message)`. This works perfectly for CLI - just log the message and update progress bar.

3. **Neural Model Memory**: `NeuralPixelizer` holds PyTorch models. For batch processing, create once, pass as parameter, reuse across all files.

4. **Config Paths**: All paths in JSON should be relative to JSON file location OR absolute. Need to resolve paths correctly.

5. **Palette from Image**: If palette source is "file:image.png", need to extract palette using `ColorReducer.generate_kmeans_palette()` with target color count.

6. **Threading vs Multiprocessing**: GUI uses threading (for status updates). CLI can use multiprocessing for image batches since no shared GUI state.

7. **FFmpeg Errors**: Video processing can fail if FFmpeg not in PATH. Catch and log clearly.

8. **JSON Comments**: JSON doesn't support comments. For documented configs, either:
   - Use companion `.md` files
   - Or add a `"_comment"` field (ignored by code)

---

## Quick Start After Plan Approval

```bash
# 1. Create feature branch
git checkout -b feature/cli-mode

# 2. Install dependencies
pip install rich

# 3. Start with Phase 1.1
# Create dither_pie_gui.py and move DitheringApp class

# 4. Work through checklist sequentially
# Test after each phase

# 5. When done, merge to main
git checkout main
git merge feature/cli-mode
```

---

**Ready to start? Begin with Phase 1.1! 🚀**

