import numpy as np
import pyrender
import trimesh
import os
from PIL import Image
import numpy as np
import os.path as osp
from tqdm import tqdm
import argparse
from src.utils.trimesh_utils import as_mesh
from src.utils.trimesh_utils import get_obj_diameter

# environment variables
os.environ["DISPLAY"] = ":1"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def process_and_resize_image(rgb_array, target_size=150):
    """
    Crop the rendered object and centre it on a square canvas:
    1. scan every row/column to find the pixel extent (top-left, bottom-right).
    2. build a square from the longest edge.
    3. centre the object and resize to `target_size`.
    """
    # use the alpha channel with a very low threshold to ignore stray pixels
    # raise 1 to 10 or 30 if faint edges still survive
    mask = rgb_array[:, :, 3] > 1

    # 1. scan rows and columns;
    # np.any(mask, axis=1) is a boolean array telling whether each row has any pixel
    rows_with_pixels = np.any(mask, axis=1)
    cols_with_pixels = np.any(mask, axis=0)

    # empty render (degenerate case)
    if not np.any(rows_with_pixels):
        return Image.new("RGB", (target_size, target_size), (255, 255, 255))

    # first and last row containing pixels
    y_min, y_max = np.where(rows_with_pixels)[0][[0, -1]]
    # first and last column containing pixels
    x_min, x_max = np.where(cols_with_pixels)[0][[0, -1]]
    
    # +1 makes the maxima inclusive
    y_max += 1
    x_max += 1

    # 2. crop the tight rectangle around the object;
    # RGB channels are kept for display, alpha is reused as the paste mask
    object_crop = Image.fromarray(rgb_array[y_min:y_max, x_min:x_max, :])
    
    # crop width and height
    h = y_max - y_min
    w = x_max - x_min
    
    # 3. square side = longest edge, so the object is never distorted
    side_length = max(h, w)

    # 4. white square canvas
    final_square = Image.new("RGB", (side_length, side_length), (255, 255, 255))

    # 5. centring offsets (the longest edge gets offset 0)
    offset_x = (side_length - w) // 2
    offset_y = (side_length - h) // 2

    # 6. paste using the alpha channel so the transparent background keeps the canvas white
    final_square.paste(object_crop, (offset_x, offset_y), mask=object_crop.split()[3])

    # 7. resize to the final size
    final_img = final_square.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return final_img

def render(
    mesh,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    light_itensity=0.6,
    is_tless=False,
    re_center_transform=np.eye(4),
):
    # camera pose is fixed as np.eye(4)
    cam_pose = np.eye(4)
    # convert openCV camera
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1
    
    # scene lighting
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])
    if light_itensity != 0.6:
        ambient_light = np.array([1.0, 1.0, 1.0, 1.0])
    
    # fully transparent background so the alpha channel can serve as a mask
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=ambient_light
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=light_itensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=cam_pose)

    # camera
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
    )
    scene.add(camera, pose=cam_pose)
    render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

    for idx_frame in tqdm(range(obj_poses.shape[0]), desc="Rendering"):
        scene.set_pose(cad_node, obj_poses[idx_frame] @ re_center_transform)
        
        # render RGBA
        rgb, depth = render_engine.render(scene, pyrender.constants.RenderFlags.RGBA)
        
        # --- post-process ---
        final_img = process_and_resize_image(rgb, target_size=150)


        final_img.save(osp.join(output_dir, f"{idx_frame:06d}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    parser.add_argument("light_itensity", nargs="?", type=float, default=0.6, help="Light itensity")
    parser.add_argument("radius", nargs="?", type=float, default=1, help="Distance from camera to object")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = "output_renders"

    is_hot3d = "hot3d" in args.output_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_devices if args.gpus_devices else "0"
    
    poses = np.load(args.obj_pose)
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    if args.radius != 1:
        poses[:, :3, 3] = poses[:, :3, 3] * args.radius
        
    if "tless" in args.output_dir:
        intrinsic = np.asarray(
            [1075.65091572, 0.0, 360, 0.0, 1073.90347929, 270, 0.0, 0.0, 1.0]
        ).reshape(3, 3)
        img_size = [540, 720]
        is_tless = True
    else:
        intrinsic = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        )
        img_size = [480, 640]
        is_tless = False

    mesh = trimesh.load_mesh(args.cad_path)
    re_center_transform = np.eye(4)
    if mesh.bounding_box is not None:
        re_center_transform[:3, 3] = -mesh.bounding_box.centroid
    
    diameter = get_obj_diameter(args.cad_path)
    if not is_hot3d and diameter > 100: 
        mesh.apply_scale(0.001)

    if is_tless:
        color = 0.4
        if hasattr(mesh, 'visual'):
             if hasattr(mesh.visual, 'face_colors'):
                mesh.visual.face_colors = np.ones((len(mesh.faces), 3)) * color
             if hasattr(mesh.visual, 'vertex_colors'):
                mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 3)) * color
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    elif is_hot3d:
        if isinstance(mesh, trimesh.Scene):
             geom = list(mesh.geometry.values())[0]
        else:
             geom = mesh
        mesh = pyrender.Mesh.from_trimesh(geom)
        if "obj_000023" not in args.cad_path and hasattr(mesh.primitives[0], 'positions'):
            mesh.primitives[0].positions = mesh.primitives[0].positions * 0.001
    else:
        mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh))
        
    os.makedirs(args.output_dir, exist_ok=True)
    render(
        output_dir=args.output_dir,
        mesh=mesh,
        obj_poses=poses,
        intrinsic=intrinsic,
        img_size=img_size, # size defined above
        light_itensity=args.light_itensity,
        re_center_transform=re_center_transform,
    )