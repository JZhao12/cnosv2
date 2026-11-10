#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render an RGB view and a coordinate (NOCS) view of a CAD model.
"""

import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def render_mesh(mesh, scale, render_type="rgb"):

    if render_type == "nocs":
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    else:
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[1.0, 1.0, 1.0])

    pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=1)
    camera_dist = scale * 1.5 
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, camera_dist], 
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    if render_type == "rgb":
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, depth = r.render(scene)
    r.delete()
    return color


def process_and_render(obj_path, angle):

    mesh = trimesh.load(obj_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    vertices = mesh.vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    scale = np.max(bbox_max - bbox_min)
    center = (bbox_min + bbox_max) / 2

    nocs_coords = (vertices - center) / scale 
    nocs_colors = np.clip(nocs_coords + 0.5, 0, 1)
    vertex_colors = (nocs_colors * 255).astype(np.uint8)

    rot_x = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    final_matrix = np.dot(rot_y, rot_x)
    
    mesh.apply_transform(final_matrix)
    mesh.vertices -= mesh.centroid

    img_rgb = render_mesh(mesh.copy(), scale, render_type="rgb")

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    img_nocs = render_mesh(mesh, scale, render_type="nocs")

    return img_rgb, img_nocs


# np.pi, np.pi/4*3, np.pi/2, np.pi/4, 0, -np.pi/4, -np.pi/2, -np.pi/4*3

if __name__ == "__main__":
    obj_path = "./datasets/bop23_challenge/datasets/ycbv/models/obj_000009.ply"    
    rotation_angle = np.pi
    view_id = 0
    save_dir = "patent"

    if os.path.exists(obj_path):
        try:
            img_rgb, img_nocs = process_and_render(obj_path, rotation_angle)
            
            Image.fromarray(img_rgb).save(os.path.join(save_dir, f"obj09_view{view_id:02d}.png"))
            Image.fromarray(img_nocs).save(os.path.join(save_dir, f"obj09_coord{view_id:02d}.png"))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(img_rgb); ax1.axis('off'); ax1.set_title("RGB")
            ax2.imshow(img_nocs); ax2.axis('off'); ax2.set_title("Coordinate")
            plt.show()
            plt.close()
            
            print(f"Rendered successfully, saved to {save_dir}/")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"File not found: {obj_path}")