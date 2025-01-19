import argparse
import os
import subprocess
import json
from tqdm import tqdm
from typing import List, Tuple

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex code to RGB tuple."""
    hex_code = hex_code.lstrip('#')
    if len(hex_code) != 6:
        raise ValueError(f"Invalid hex code: {hex_code}")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def load_custom_palettes(palette_file: str) -> List[Tuple[str, List[str]]]:
    """Load custom palettes from palette.json (name plus list of hex colors)."""
    custom_palettes = []
    if os.path.exists(palette_file):
        try:
            with open(palette_file, 'r') as f:
                data = json.load(f)
            for palette in data:
                name = palette['name']
                colors = palette['colors']  # List of hex codes
                custom_palettes.append((name, colors))
        except Exception as e:
            print(f"Warning: Failed to load custom palettes from '{palette_file}': {e}")
    else:
        print(f"Info: '{palette_file}' not found. Proceeding without custom palettes.")
    return custom_palettes

def get_image_paths(input_path: str) -> List[str]:
    """Retrieve image file paths from a directory or single file."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Supported image extensions
        supported_ext = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        image_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(supported_ext)
        ]
        return image_files
    else:
        raise FileNotFoundError(f"Input path '{input_path}' is neither a file nor a directory.")

def create_output_dir(base_output_dir: str) -> str:
    """Create the main output directory."""
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing or removing invalid characters."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for ch in invalid_chars:
        filename = filename.replace(ch, '_')
    return filename

def main():
    parser = argparse.ArgumentParser(
        description="Batch Image Dithering and Pixelization Tool using the new dither_pie.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help="Path to the input image file or directory containing images."
    )
    args = parser.parse_args()
    
    # Load custom palettes (from palette.json)
    palette_file = 'palette.json'
    custom_palettes = load_custom_palettes(palette_file)
    
    # ----------------------------------------------------------------------
    # 1) Define sets of palettes for the new CLI usage
    #
    # We separate these into:
    #   - Algorithmic palettes: pass via `--algo-palette <value>`
    #   - Custom palettes: pass via `-p <custom_palette_name>`
    #
    # The numbers given below (16) are simply the color count we feed to -c.
    # ----------------------------------------------------------------------

    # Algorithmic (built-in) palette "names" that map to our new `--algo-palette` usage:
    algo_palettes = [
        ("median_cut", 16),
        ("kmeans_variant1", 16),
        ("kmeans_variant2", 16),
        ("uniform", 16),
    ]

    # Custom palettes (loaded from palette.json). We'll also feed -c 16, but the user might have more or fewer colors in them. 
    # The second item in the tuple is the "number of colors" we pass to the CLI (just a default).
    # The actual color list is in memory, but we only need the name for -p.
    # e.g., ("gb_dmg_palette", ["#0f381f", "#304e2a", ...])
    # We'll keep the color count as 16 for each. You could tune it if you want.
    custom_palettes_for_cli = [(name, 16) for (name, _) in custom_palettes]
    
    # Combine dithering modes
    dithering_modes = ["none", "bayer2x2", "bayer4x4", "bayer8x8", "bayer16x16"]
    
    # Get list of image paths
    try:
        image_paths = get_image_paths(args.input_path)
        if not image_paths:
            print(f"No supported image files found in '{args.input_path}'.")
            return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create output directory
    base_output_dir = 'processed_results'
    output_dir = create_output_dir(base_output_dir)
    
    # Path to the dither_pie.py script
    cli_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dither_pie.py')
    if not os.path.exists(cli_script):
        print(f"Error: '{cli_script}' not found. Ensure that 'dither_pie.py' is in the same directory.")
        return
    
    # Prepare list of tasks
    tasks = []
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_image_name = sanitize_filename(image_name)
        
        # 1. Pixelize (simple: pixelize command, -m 640)
        pixelized_output = os.path.join(output_dir, f"{sanitized_image_name}_pixelized.png")
        tasks.append({
            'type': 'pixelize',
            'args': [
                'pixelize',
                image_path,
                pixelized_output,
                '-m', '640'  # e.g., max_size
            ]
        })
        
        # 2. Dither with each dithering mode, for each *algorithmic palette*
        for mode in dithering_modes:
            for (algo_name, color_count) in algo_palettes:
                # Example:
                #   python dither_pie.py dither input_image output_image
                #       -d bayer4x4 -c 16 --algo-palette kmeans_variant1
                dithered_output = os.path.join(
                    output_dir,
                    f"{sanitized_image_name}_dithered_{mode}_{sanitize_filename(algo_name)}.png"
                )
                tasks.append({
                    'type': 'dither',
                    'args': [
                        'dither',
                        image_path,
                        dithered_output,
                        '-d', mode,
                        '-c', str(color_count),
                        '--algo-palette', algo_name
                    ]
                })

        # 2b. Dither with each dithering mode, for each *custom palette*
        for mode in dithering_modes:
            for (cust_name, color_count) in custom_palettes_for_cli:
                # Example:
                #   python dither_pie.py dither input_image output_image
                #       -d bayer4x4 -c 16 -p gb_dmg_palette
                dithered_output = os.path.join(
                    output_dir,
                    f"{sanitized_image_name}_dithered_{mode}_{sanitize_filename(cust_name)}.png"
                )
                tasks.append({
                    'type': 'dither',
                    'args': [
                        'dither',
                        image_path,
                        dithered_output,
                        '-d', mode,
                        '-c', str(color_count),
                        '-p', cust_name
                    ]
                })
        
        # 3. Dither-Pixelize (both algorithmic and custom)
        #    same approach, but use 'dither-pixelize' command
        for mode in dithering_modes:
            for (algo_name, color_count) in algo_palettes:
                dither_pixelized_output = os.path.join(
                    output_dir,
                    f"{sanitized_image_name}_dither_pixelized_{mode}_{sanitize_filename(algo_name)}.png"
                )
                tasks.append({
                    'type': 'dither-pixelize',
                    'args': [
                        'dither-pixelize',
                        image_path,
                        dither_pixelized_output,
                        '-d', mode,
                        '-c', str(color_count),
                        '--algo-palette', algo_name,
                        '-m', '640'
                    ]
                })
            for (cust_name, color_count) in custom_palettes_for_cli:
                dither_pixelized_output = os.path.join(
                    output_dir,
                    f"{sanitized_image_name}_dither_pixelized_{mode}_{sanitize_filename(cust_name)}.png"
                )
                tasks.append({
                    'type': 'dither-pixelize',
                    'args': [
                        'dither-pixelize',
                        image_path,
                        dither_pixelized_output,
                        '-d', mode,
                        '-c', str(color_count),
                        '-p', cust_name,
                        '-m', '640'
                    ]
                })
    
    # Run tasks with tqdm progress bar
    with tqdm(total=len(tasks), desc="Processing Tasks") as pbar:
        for task in tasks:
            try:
                subprocess.run([
                    'python', cli_script
                ] + task['args'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error during {task['type']} for '{task['args'][1]}': {e}")
            pbar.update(1)
    
    print(f"\nAll processing complete. Results saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
