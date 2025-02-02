import sys
import os
import cv2
import ffmpeg
from PIL import Image

def resize_image(input_path, target_size):
    """Resizes an image while keeping aspect ratio using nearest-neighbor scaling."""
    img = Image.open(input_path)
    img = img.convert("RGB")  # Ensures compatibility

    # Get original dimensions
    width, height = img.size

    # Determine scaling factor based on the smallest side
    scale_factor = target_size / min(width, height)
    new_width = round(width * scale_factor)
    new_height = round(height * scale_factor)

    # Resize using nearest-neighbor
    resized_img = img.resize((new_width, new_height), Image.NEAREST)

    # Save with suffix
    output_path = f"{os.path.splitext(input_path)[0]}_resized{os.path.splitext(input_path)[1]}"
    resized_img.save(output_path)
    print(f"Image saved: {output_path}")

def resize_video(input_path, target_size):
    """Resizes a video while keeping aspect ratio, ensuring dimensions are even."""
    probe = ffmpeg.probe(input_path)
    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)

    if not video_stream:
        print("No video stream found in the file.")
        return

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # Compute new dimensions while maintaining aspect ratio
    scale_factor = target_size / min(width, height)
    new_width = int(round(width * scale_factor / 2) * 2)  # Ensure even width
    new_height = int(round(height * scale_factor / 2) * 2)  # Ensure even height

    output_path = f"{os.path.splitext(input_path)[0]}_resized{os.path.splitext(input_path)[1]}"

    # Resize video while preserving all streams
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            vf=f"scale={new_width}:{new_height}:flags=neighbor",  # Nearest-neighbor scaling
            vcodec="libx264",  # Re-encode video using H.264
            preset="fast",      # Faster encoding
            acodec="copy",      # Copy audio without re-encoding
            scodec="copy",      # Copy subtitles without re-encoding
            map="0"             # Preserve all streams
        )
        .run(overwrite_output=True)
    )

    print(f"Video saved: {output_path}")



def main():
    if len(sys.argv) != 3:
        print("Usage: python resizer.py NUMBER path-to-file.img/vid")
        sys.exit(1)

    try:
        target_size = int(sys.argv[1])
    except ValueError:
        print("Error: NUMBER must be an integer.")
        sys.exit(1)

    input_path = sys.argv[2]

    if not os.path.exists(input_path):
        print("Error: File not found.")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        resize_image(input_path, target_size)
    elif ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        resize_video(input_path, target_size)
    else:
        print("Unsupported file type.")
        sys.exit(1)

if __name__ == "__main__":
    main()
