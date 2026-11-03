#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stitch rendered views into a single grid image.
"""

import os
from PIL import Image

def grid_stitch_images(image_dir, output_filename="stitched_view.png", rows=6, cols=7, gap=10):
    """
    Stitch the images of a folder into one grid image.
    :param image_dir: folder containing the images
    :param output_filename: name of the output file
    :param rows: number of rows (6)
    :param cols: number of columns (7)
    :param gap: gap between images, in pixels
    """
    # 1. build the file list (000000.png .. 000041.png)
    # range is more robust than a sorted os.listdir because a missing file is detected explicitly
    total_images = rows * cols
    image_names = [f"{i:06d}.png" for i in range(total_images)]
    
    # check the first image exists and read its size
    first_img_path = os.path.join(image_dir, image_names[0])
    if not os.path.exists(first_img_path):
        print(f"Error: first image not found: {first_img_path}")
        return

    with Image.open(first_img_path) as img:
        w, h = img.size
    
    print(f"Single image size: {w}x{h}")

    # 2. total canvas size
    # width  = cols * w + (cols - 1) * gap
    # height = rows * h + (rows - 1) * gap
    # (add 2 * gap here if you want an outer margin)
    total_w = cols * w + (cols - 1) * gap
    total_h = rows * h + (rows - 1) * gap

    # 3. white canvas
    new_image = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    # 4. paste every image
    print("Stitching...")
    for idx, name in enumerate(image_names):
        img_path = os.path.join(image_dir, name)
        
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {name}")
            continue
            
        try:
            with Image.open(img_path) as img:
                # row and column of this image
                row_idx = idx // cols
                col_idx = idx % cols
                
                # paste position
                x = col_idx * (w + gap)
                y = row_idx * (h + gap)
                
                new_image.paste(img, (x, y))
        except Exception as e:
            print(f"Failed to read {name}: {e}")

    # 5. save the result
    output_path = os.path.join(image_dir, output_filename)
    new_image.save(output_path)
    print(f"Stitched image saved to {output_path}")

if __name__ == "__main__":
    # configuration
    target_dir = "./tmp/ycbv_models/obj_000015"
    
    # 6 rows x 7 columns, 15 px gap (adjust as needed)
    grid_stitch_images(target_dir, rows=6, cols=7, gap=15)