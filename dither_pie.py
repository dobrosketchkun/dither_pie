#!/usr/bin/env python3
"""
Dither Pie - Entry Point

Routes to GUI or CLI mode based on command-line arguments:
- No arguments: Launch GUI
- Arguments present: Launch CLI
"""

import sys


def main():
    """Main entry point - routes to GUI or CLI."""
    if len(sys.argv) == 1:
        # No arguments - launch GUI
        from dither_pie_gui import launch_gui
        launch_gui()
    else:
        # Arguments present - launch CLI
        from dither_cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
