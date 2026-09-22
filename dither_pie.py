#!/usr/bin/env python3
"""
Dither Pie - Entry Point

Routes to GUI or CLI mode based on command-line arguments:
- No arguments: Launch GUI
- Arguments present: Launch CLI
"""

import sys

__all__ = ['main']


def main():
    """Main entry point - routes to GUI or CLI."""
    gui_flags = {'--old-design', '--old_design', '--new-design', '--new_design'}
    requested = gui_flags.intersection(sys.argv[1:])
    if requested:
        if len(sys.argv) != 2:
            raise SystemExit('GUI design flags must be used alone; CLI processing options are separate.')
        if sys.argv[1] in {'--new-design', '--new_design'}:
            from ui.modern.app import launch_gui
        else:
            from dither_pie_gui import launch_gui
        launch_gui()
    elif len(sys.argv) == 1:
        # No arguments - launch GUI
        from dither_pie_gui import launch_gui
        launch_gui()
    else:
        # Arguments present - launch CLI
        from dither_cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
