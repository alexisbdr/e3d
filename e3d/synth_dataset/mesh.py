import os
import random
from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import TexturesVertex, look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (Rotate, RotateAxisAngle, Transform3d,
                                  Translate)

# Don't need this
rotation_dict = {
    "bottle_square": [90, 0, 180],
    "sofa": [0, 0, 180],
    "plane_commercial": [90, 0, 180],
    "plane_fighter": [90, 0, 180],
    "chair_plastic": [0, 90, 90],
    "cat": [0, 0, 0],
    "car_beetle": [90, 0, 180],
    "hand": [0, 0, 20],
    "bottle_beer": [-90, 0, 0],
    "horse": [-90, 90, 0],
    "car_fiat": [0, 90, -2],
    "teapot": [0, 0, 0],
    "lamp": [0, 0, 0],
    "bottle_spray": [90, 0, 180],
    "dolphin": [0, 0, 0],
    "plane_WWII": [90, 0, 0],
    "body_male": [0, 0, 0],
    "helicopter": [0, 0, 0],
    "head_female": [90, 0, 180],
    "schoolbus": [0, 90, 90],
    "chair_foldout": [-90, 0, 0],
}


def scale_mesh(mesh):
    """Scale normalize the input mesh to be centered at (0,0,0) and fits in a sphere of 1
    """
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center.expand(N, 3))
    mesh.scale_verts_((1.0 / float(scale)))

    return mesh


def mesh_random_translation(mesh, bound: float, device: str = ""):
    """
        Generates a random translation for the mesh.
        Input:
            -mesh: Pytorch3d meshes
            -bound: translation in pixel units
        Returns the altered mesh
    """

    upper = bound
    lower = -1 * bound

    t_params_x = round(random.uniform(lower, upper), 2)
    # Y bounds logic
    y_upper = min(upper, t_params_x) if t_params_x >= 0 else upper
    y_lower = max(lower, t_params_x) if t_params_x < 0 else lower
    t_params_y = round(random.uniform(y_lower, y_upper), 2)
    # Z bounds logic
    z_upper = min(upper, max(t_params_x, t_params_y)) if t_params_y >= 0 else upper
    z_lower = min(upper, max(t_params_x, t_params_y)) if t_params_y < 0 else lower
    t_params_z = round(random.uniform(z_lower, z_upper), 2)

    transform = Transform3d(device=device).translate(t_params_x, t_params_y, t_params_z)
    verts, faces = mesh.get_mesh_verts_faces(0)
    verts = transform.transform_points(verts)
    mesh = mesh.update_padded(verts.unsqueeze(0))

    return mesh, transform


def mesh_random_rotation(mesh, bound: int):
    """
    """
    return


def rotate_mesh_around_axis(
    mesh, rot: list, renderer, dist: float = 3.5, save: str = "", device: str = ""
):

    if not device:
        device = torch.cuda.current_device()

    rot_x = RotateAxisAngle(rot[0], "X", device=device)
    rot_y = RotateAxisAngle(rot[1], "Y", device=device)
    rot_z = RotateAxisAngle(rot[2], "Z", device=device)

    rot = Transform3d(device=device).stack(rot_x, rot_y, rot_z)

    verts, faces = mesh.get_mesh_verts_faces(0)
    verts = rot_x.transform_points(verts)
    verts = rot_y.transform_points(verts)
    verts = rot_z.transform_points(verts)
    mesh = mesh.update_padded(verts.unsqueeze(0))

    dist = dist
    elev = 0
    azim = 0

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)

    image_ref = renderer(meshes_world=mesh, R=R, T=T, device=device)
    image_ref = image_ref.cpu().numpy()[..., :3]

    plt.imshow(image_ref.squeeze())
    plt.show()

    if save:
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_obj(save, verts, faces)
    return mesh


def translate_mesh_on_axis(
    mesh, t: list, renderer, dist: float = 3.5, save: str = "", device: str = ""
):

    translation = Transform3d(device=device).translate(t[0], t[1], t[2])

    verts, faces = mesh.get_mesh_verts_faces(0)
    verts = translation.transform_points(verts)
    mesh = mesh.update_padded(verts.unsqueeze(0))

    dist = dist
    elev = 0
    azim = 0

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)

    image_ref = renderer(meshes_world=mesh, R=R, T=T, device=device)
    image_ref = image_ref.cpu().numpy()

    plt.imshow(image_ref.squeeze())
    plt.show()

    if save:
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_obj(save, verts, faces)
    return mesh


def load_meshes(meshes_path: str = "./data/meshes", device: str = "") -> dict:

    meshes_path = abspath(join(dirname(__file__), meshes_path))
    meshes_files = [
        join(meshes_path, mesh)
        for mesh in os.listdir(meshes_path)
        if mesh.endswith(".obj")
    ]

    # Load the object without textures and materials
    meshes = {}
    for path in meshes_files:
        verts, faces_idx, aux = load_obj(path)
        faces = faces_idx.verts_idx

        # Scale normalize the target mesh to fit in a sphere of radius 1 centered at (0,0,0)
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        # Initialize each vertex to be white in color.
        textures = aux.texture_atlas
        if textures:
            textures = verts.new_ones(faces.shape[0], 4, 4, 3)
            textures = TexturesVertex(verts_features=verts.to(device))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        mesh_name = path.split("/")[-1].split(".")[0]
        meshes[mesh_name] = Meshes(
            verts=[verts.to(device)], faces=[faces.to(device)], textures=textures
        )

    return meshes
