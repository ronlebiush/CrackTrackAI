import argparse
import os

import cv2
import numpy as np


def split_image(image_path, output_dir, tile_size, exclude_bar_height):
    """
    Split a single SEM image into smaller tiles and save them to the output directory,
    excluding the black bar at the bottom of the image.

    :param image_path: Path to the input SEM image.
    :param output_dir: Directory where the tiles will be saved.
    :param tile_size: Size of each tile (e.g., 256 for 256x256 tiles).
    :param exclude_bar_height: Height of the black bar at the bottom to exclude (in pixels).
    """
    # Load the SEM image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Cannot load image {image_path}. Skipping...")
        return

    # Get image dimensions
    img_height, img_width = image.shape

    # Exclude the black bar
    cropped_height = img_height - exclude_bar_height
    if cropped_height <= 0:
        print(
            f"Warning: Image {image_path} is too small after excluding the black bar. Skipping..."
        )
        return

    image = image[:cropped_height, :]

    # Get the base name of the image file (without extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Split the image into tiles
    tile_count = 0
    for y in range(0, cropped_height, tile_size):
        for x in range(0, img_width, tile_size):
            # Extract the tile
            tile = image[y : y + tile_size, x : x + tile_size]

            # Check if the tile is the desired size (to handle edge cases)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            # Save the tile
            tile_filename = os.path.join(
                output_dir, f"{base_name}_tile_{tile_count}.png"
            )
            cv2.imwrite(tile_filename, tile)
            tile_count += 1

    print(f"Processed {image_path}: {tile_count} tiles saved.")


def process_folder(input_folder, output_dir, tile_size, exclude_bar_height):
    """
    Process all images in a folder, splitting each into smaller tiles while excluding the black bar.

    :param input_folder: Path to the folder containing input images.
    :param output_dir: Directory where the tiles will be saved.
    :param tile_size: Size of each tile (e.g., 256 for 256x256 tiles).
    :param exclude_bar_height: Height of the black bar at the bottom to exclude (in pixels).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image (by extension)
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".bmp")
        ):
            split_image(file_path, output_dir, tile_size, exclude_bar_height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split SEM images in a folder into smaller tiles, excluding the black bar."
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Directory to save the tiles."
    )
    parser.add_argument(
        "-t",
        "--tile_size",
        type=int,
        default=256,
        help="Size of each tile (default: 256).",
    )
    parser.add_argument(
        "-b",
        "--exclude_bar_height",
        type=int,
        default=50,
        help="Height of the black bar at the bottom to exclude (default: 50).",
    )

    args = parser.parse_args()

    # Run the process_folder function
    process_folder(
        args.input_folder, args.output, args.tile_size, args.exclude_bar_height
    )
