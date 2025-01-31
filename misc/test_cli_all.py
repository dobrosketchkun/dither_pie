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
    """Load custom palettes from palette.json (name plus list of hex codes)."""
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

def is_video_file(path: str) -> bool:
    """Check if a file is a recognized video extension."""
    video_exts = ('.mp4', '.mkv', '.avi', '.mov')
    return os.path.isfile(path) and path.lower().endswith(video_exts)

def is_image_file(path: str) -> bool:
    """Check if a file is recognized image extension."""
    img_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    return os.path.isfile(path) and path.lower().endswith(img_exts)

def get_image_or_video_paths(input_path: str) -> List[str]:
    """
    Retrieve file paths from a directory or single file.
    Gathers images or video files if in directory,
    or just returns [input_path] if single file.
    """
    if os.path.isfile(input_path):
        # Single file. Check if it's either image or video:
        if is_video_file(input_path) or is_image_file(input_path):
            return [input_path]
        else:
            raise FileNotFoundError(
                f"Input path '{input_path}' is neither a recognized image nor video extension."
            )
    elif os.path.isdir(input_path):
        # Collect both images and videos
        supported_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.mkv', '.avi', '.mov')
        file_list = []
        for f in os.listdir(input_path):
            full = os.path.join(input_path, f)
            if os.path.isfile(full) and f.lower().endswith(supported_exts):
                file_list.append(full)
        return file_list
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
        description="Batch Image/Video Dithering and Pixelization Tool using dither_pie.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help="Path to a single image/video file or a directory containing images/videos."
    )
    args = parser.parse_args()
    
    # Load custom palettes (from palette.json)
    palette_file = 'palette.json'
    custom_palettes = load_custom_palettes(palette_file)
    
    # Algorithmic (built-in) palette "names"
    algo_palettes = [
        ("median_cut", 16),
        ("kmeans_variant1", 16),
        ("kmeans_variant2", 16),
        ("uniform", 16),
    ]
    # Custom palettes: just name + color_count = 16 (arbitrary default)
    custom_palettes_for_cli = [(name, 16) for (name, _) in custom_palettes]
    # Dithering modes
    dithering_modes = ["none", "bayer2x2", "bayer4x4", "bayer8x8", "bayer16x16"]
    
    # Gather files
    try:
        paths = get_image_or_video_paths(args.input_path)
        if not paths:
            print(f"No supported image or video files found in '{args.input_path}'.")
            return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Create an output directory
    base_output_dir = 'processed_results'
    output_dir = create_output_dir(base_output_dir)
    
    # Path to the dither_pie.py
    cli_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dither_pie.py')
    if not os.path.exists(cli_script):
        print(f"Error: '{cli_script}' not found. Ensure that 'dither_pie.py' is in the same directory.")
        return
    
    tasks = []
    for path in paths:
        # Determine if image or video
        is_video = is_video_file(path)
        extension = ".mp4" if is_video else ".png"
        
        # Get base name for outputs
        name = os.path.splitext(os.path.basename(path))[0]
        sname = sanitize_filename(name)
        
        # 1. Pixelize
        pixelized_output = os.path.join(output_dir, f"{sname}_pixelized{extension}")
        tasks.append({
            'type': 'pixelize',
            'args': [
                'pixelize',
                path,
                pixelized_output,
                '-m', '640'
            ]
        })
        
        # 2. Dither (algorithmic + custom)
        for mode in dithering_modes:
            # Algorithmic
            for (algo_name, color_count) in algo_palettes:
                outpath = os.path.join(output_dir,
                    f"{sname}_dithered_{mode}_{sanitize_filename(algo_name)}{extension}")
                tasks.append({
                    'type': 'dither',
                    'args': [
                        'dither',
                        path,
                        outpath,
                        '-d', mode,
                        '-c', str(color_count),
                        '--algo-palette', algo_name
                    ]
                })
            # Custom
            for (cust_name, color_count) in custom_palettes_for_cli:
                outpath = os.path.join(output_dir,
                    f"{sname}_dithered_{mode}_{sanitize_filename(cust_name)}{extension}")
                tasks.append({
                    'type': 'dither',
                    'args': [
                        'dither',
                        path,
                        outpath,
                        '-d', mode,
                        '-c', str(color_count),
                        '-p', cust_name
                    ]
                })
        
        # 3. dither-pixelize (algorithmic + custom)
        for mode in dithering_modes:
            # Algorithmic
            for (algo_name, color_count) in algo_palettes:
                outpath = os.path.join(output_dir,
                    f"{sname}_dither_pixelized_{mode}_{sanitize_filename(algo_name)}{extension}")
                tasks.append({
                    'type': 'dither-pixelize',
                    'args': [
                        'dither-pixelize',
                        path,
                        outpath,
                        '-d', mode,
                        '-c', str(color_count),
                        '--algo-palette', algo_name,
                        '-m', '640'
                    ]
                })
            # Custom
            for (cust_name, color_count) in custom_palettes_for_cli:
                outpath = os.path.join(output_dir,
                    f"{sname}_dither_pixelized_{mode}_{sanitize_filename(cust_name)}{extension}")
                tasks.append({
                    'type': 'dither-pixelize',
                    'args': [
                        'dither-pixelize',
                        path,
                        outpath,
                        '-d', mode,
                        '-c', str(color_count),
                        '-p', cust_name,
                        '-m', '640'
                    ]
                })
    
    # Run tasks with a progress bar
    with tqdm(total=len(tasks), desc="Processing Tasks") as pbar:
        for task in tasks:
            try:
                subprocess.run(
                    ['python', cli_script] + task['args'],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"\nError during {task['type']} for '{task['args'][1]}': {e}")
            pbar.update(1)
    
    print(f"\nAll processing complete. Results saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
