import subprocess
import argparse
import sys
import os

def run_ffmpeg_command(command, description):
    """A helper function to run an FFmpeg command and print its output."""
    print(f"\n--- {description} ---")
    print(' '.join(command))
    print("--------------------------------" + "-" * len(description) + "\n")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg command failed with exit code {process.returncode}")
        return True
    except FileNotFoundError:
        print("❌ Error! `ffmpeg` command not found.")
        print("Please make sure FFmpeg is installed and accessible in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def combine_videos(video1_path, video2_path, output_path, orientation):
    """
    Combines two videos horizontally or vertically using a robust two-stage process.
    """
    temp_video1_path = "temp_sanitized_1.mp4"
    temp_video2_path = "temp_sanitized_2.mp4"
    
    # --- STAGE 1: Sanitize both videos into temporary files ---
    sanitize_command_1 = [
        'ffmpeg', '-y', '-i', video1_path,
        '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast',
        '-c:a', 'copy', '-r', '24000/1001',
        temp_video1_path
    ]
    
    sanitize_command_2 = [
        'ffmpeg', '-y', '-i', video2_path,
        '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast',
        '-c:a', 'copy', '-r', '24000/1001',
        temp_video2_path
    ]

    # --- Build the filter for the final combination step ---
    if orientation == 'h':
        # Scale video2's height to match video1's height, preserving sharp pixels.
        # Then, stack them side-by-side.
        filter_complex = (
            "[1:v]scale=-1:ih:flags=neighbor[right];"
            "[0:v][right]hstack=inputs=2[v];"
            "[0:a][1:a]amerge=inputs=2,pan=stereo|c0<c0+c2|c1<c1+c3[a]"
        )
        stage2_description = "Stage 2/2: Combining videos (Horizontal)"
    else: # orientation == 'v'
        # Scale video2's width to match video1's width, preserving sharp pixels.
        # Then, stack them top-to-bottom.
        filter_complex = (
            "[1:v]scale=iw:-1:flags=neighbor[bottom];"
            "[0:v][bottom]vstack=inputs=2[v];"
            "[0:a][1:a]amerge=inputs=2,pan=stereo|c0<c0+c2|c1<c1+c3[a]"
        )
        stage2_description = "Stage 2/2: Combining videos (Vertical)"

    combine_command = [
        'ffmpeg', '-y',
        '-i', temp_video1_path,
        '-i', temp_video2_path,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '18',
        output_path
    ]

    try:
        if not run_ffmpeg_command(sanitize_command_1, "Stage 1/2: Sanitizing first video"):
            return
        if not run_ffmpeg_command(sanitize_command_2, "Stage 1/2: Sanitizing second video (pixel art)"):
            return
        if not run_ffmpeg_command(combine_command, stage2_description):
            return
            
        print(f"\n✅ Success! Final video saved to {output_path}")

    finally:
        # --- STAGE 3: Cleanup ---
        print("\n--- Cleaning up temporary files ---")
        for temp_file in [temp_video1_path, temp_video2_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed {temp_file}")
        print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustly combine two videos, preserving pixel art quality on the second video.")
    parser.add_argument("input1", help="Path to the first (left/top) video file.")
    parser.add_argument("input2", help="Path to the second (right/bottom, pixel art) video file.")
    parser.add_argument("output", help="Path for the output video file.")
    parser.add_argument(
        '-o', '--or',
        dest='orientation',
        default='h',
        choices=['h', 'v'],
        help="Stacking orientation: 'h' for horizontal or 'v' for vertical. Default: 'h'."
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.input1):
        print(f"Error: Input file not found at {args.input1}")
    elif not os.path.exists(args.input2):
        print(f"Error: Input file not found at {args.input2}")
    else:
        combine_videos(args.input1, args.input2, args.output, args.orientation)