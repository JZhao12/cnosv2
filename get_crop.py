import os
import numpy as np
from PIL import Image

def process_object_square(rgb_path, mask_path, output_path, final_size=150):
    """
    1. Cut the object out of the RGB image using the mask.
    2. Crop it to its tight bounding box.
    3. Place the crop in the centre of a square canvas whose side is the longest edge.
    4. Resize to `final_size` (150x150 by default) and save.
    """
    # --- 1. Load and preprocess ---
    if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
        print(f"Error: file not found\nRGB: {rgb_path}\nMask: {mask_path}")
        return

    rgb_img = Image.open(rgb_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")  # convert to grayscale

    # make sure the mask matches the image size
    if rgb_img.size != mask_img.size:
        mask_img = mask_img.resize(rgb_img.size, Image.Resampling.NEAREST)

    rgb_arr = np.array(rgb_img)
    mask_arr = np.array(mask_img)

    # binarise the mask
    binary_mask = mask_arr > 0

    # --- 2. Compute the bounding box ---
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows):
        print("Error: the mask is all black, no object found")
        return

    # bounding box coordinates
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_max += 1
    x_max += 1

    # --- 3. Mask out and crop ---
    # make the background black first
    masked_arr = np.zeros_like(rgb_arr)
    masked_arr[binary_mask] = rgb_arr[binary_mask]

    # keep only the object
    object_crop_arr = masked_arr[y_min:y_max, x_min:x_max]
    object_crop_img = Image.fromarray(object_crop_arr)

    # size of the crop
    w, h = object_crop_img.size
    print(f"Raw crop size: {w}x{h}")

    # --- 4. Centre the crop on a square canvas ---
    # side length is the longest edge
    side_length = max(w, h)
    
    # black square canvas; use "RGBA" with (0, 0, 0, 0) for a transparent background
    square_canvas = Image.new("RGB", (side_length, side_length), (0, 0, 0))

    # centring offsets
    offset_x = (side_length - w) // 2
    offset_y = (side_length - h) // 2

    # paste the object
    square_canvas.paste(object_crop_img, (offset_x, offset_y))

    # --- 5. Resize to the target size (150x150) ---
    final_img = square_canvas.resize((final_size, final_size), Image.Resampling.LANCZOS)

    # --- 6. Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path)
    print(f"Done, saved to {output_path} ({final_size}x{final_size})")

if __name__ == "__main__":
    # inputs
    rgb_file = "./media/ycbv_test_bop19/test/000049/rgb/000001.png"
    mask_file = "./media/ycbv_test_bop19/test/000049/mask_visib/000001_000002.png"
    
    # output
    output_file = "./tmp/ycbv_models/crop/obj09.png"

    process_object_square(rgb_file, mask_file, output_file)