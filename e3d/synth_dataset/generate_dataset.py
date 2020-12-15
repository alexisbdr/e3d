# imports and dependencies
import argparse
import os
import re
import sys
from os.path import abspath, dirname, join

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


sys.path.insert(0, abspath(join("..", dirname(os.getcwd()))))

import logging
import random

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mesh_reconstruction.renderer import flat_renderer, silhouette_renderer
# Plotting Libs
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageOps
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (BlendParams, HardFlatShader, HardPhongShader,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SfMPerspectiveCameras,
                                SoftPhongShader, SoftSilhouetteShader,
                                TexturesAtlas, TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from skimage import img_as_ubyte
from synth_dataset.event_renderer import generate_event_frames
from synth_dataset.mesh import (load_meshes, mesh_random_translation,
                                rotate_mesh_around_axis, scale_mesh,
                                translate_mesh_on_axis)
from synth_dataset.params import Params
from synth_dataset.trajectory import cam_trajectory
from tqdm import tqdm_notebook
from utils.manager import ImageManager, RenderManager
from utils.visualization import plot_trajectory_cameras


def get_args():

    parser = argparse.ArgumentParser(
        description="Synthetic Dataset Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name", dest="name", required=True, type=str, help="Dataset Name",
    )
    parser.add_argument(
        "--gpu", dest="gpu_num", default="0", type=str, help="GPU Device Number",
    )
    parser.add_argument(
        "--category",
        dest="category",
        default="car",
        type=str,
        help="Shapenet Category",
    )

    return parser.parse_args()


def shapenet_models(params, index: int = 0):
    """Generator of shapenet models
    """
    model_path = "models/model_normalized.obj"
    synset = params.synsets[params.category]

    model_list = os.listdir(join(params.shapenet_path, synset))
    model_paths = [
        join(params.shapenet_path, synset, c, model_path) for c in model_list
    ]
    for num, path in enumerate(model_paths):
        try:
            verts, faces, aux = load_obj(
                path, load_textures=True, create_texture_atlas=True
            )
            mesh = Meshes(
                verts=[verts],
                faces=[faces.verts_idx],
                textures=TexturesAtlas(atlas=[aux.texture_atlas]),
            ).to(device)
            print(f"Adding mesh num {num}: {model_list[num]} ")

            yield mesh, model_list[num]

        except Exception as e:
            # car_exclude_pytorch3d.append(car_list[num])
            print(e, model_list[num])
            continue


def merge_with_background(image, params, background=None, show: bool = False):
    """Utility function to composite two images:
        -image: photo-realistic image of textured mesh
        -background: python generator of textured backgrounds
    """

    def background_generator():
        files = []
        background_folder = "../data/sun360"
        files = os.listdir(background_folder)
        rand_file = random.randint(0, len(files) - 1)
        rand_file_path = abspath(join(background_folder, files[rand_file]))
        rand_file_path = abspath(
            join(background_folder, "pano_530c02959a7fdf8fdda4bac494ba3724")
        )
        files = os.listdir(rand_file_path)
        for img_num in range(len(files)):
            path_img = join(rand_file_path, f"{img_num}.jpg")
            img = Image.open(path_img).resize(params.img_size)
            yield np.array(img).astype(np.uint8)

    if background is None:
        background = background_generator()

    # Image.fromarray((np.array(image) * 255).astype(np.uint8)).save("test.png")
    # image = Image.open("test.png")

    image = img_as_ubyte(np.clip(image, 0, 1))[..., :3]

    image = np.array(image).astype(np.uint8)

    try:
        image_bg = next(background)
    except StopIteration:
        background = background_generator()
        image_bg = next(background)

    # image_thresh = (image > 1) * 255

    image_white = np.all(image == [255, 255, 255], axis=-1)
    image[image_white] = image_bg[image_white]
    # img_add = np.amin((image_bg + image), 255)
    # img_add[img_add > 255] = 255

    if show:
        plt.imshow(image)
        plt.show()

    return image, background


def main(params):
    """Synthetic dataset creation loop:
        1. loops through a batch of meshes
        2. generates a random camera trajectory
        3.
    """
    if params.shapenet:
        meshes = shapenet_models(params)
    else:
        mesh = load_objs_as_meshes(
            [params.mesh_path],
            create_texture_atlas=False,
            load_textures=True,
            device=params.device,
        )
        meshes = mesh.extend(params.mesh_iter)

    renderer = flat_renderer(params.img_size, params.device)

    for mesh in meshes:
        # Create a random trajectory
        cam_poses = cam_trajectory(
            params.variation, params.pepper, params.random_start, params.batch_size
        )

        if params.shapenet:
            mesh_id = mesh[1]
            mesh = mesh[0]
        else:
            mesh_id = params.mesh_paths

        mesh, translation = mesh_random_translation(
            mesh, params.mesh_translation, device=params.device
        )
        mesh = mesh.to(params.device)
        background = None

        # Batch indices to actually save
        data_indices = sorted(
            random.sample(range(params.batch_size), k=params.data_batch_size)
        )

        renders = dict(phong=[], silhouette=[], events=[])

        render_manager = RenderManager(
            types=list(renders.keys()),
            new_folder=f"{params.name}",
            metadata={
                "augmentation_params": {
                    "variation": params.variation,
                    "pepper": params.pepper,
                    "random_start": params.random_start,
                },
                "mesh_transformation": {
                    "translation": translation.get_matrix().cpu().numpy().tolist()
                },
                "mesh_info": {
                    "mesh_id": mesh_id,
                    "synset_id": params.synsets[params.category]
                    if params.shapenet
                    else "",
                    "category_name": params.category if params.shapenet else "",
                },
            },
        )
        render_manager.init()

        R, T = cam_poses
        data_img_num = 0
        for idx in range(1, len(R) + 1):
            img_dict = {}

            if "phong" in renders.keys():
                camera = SfMPerspectiveCameras(
                    R=R[idx - 1 : idx :], T=T[idx - 1 : idx :], device=params.device
                )
                image_ref = renderer(
                    meshes_world=mesh, cameras=camera, device=params.device
                )
                image_ref = image_ref.cpu().numpy()
                img_dict["phong"] = image_ref.squeeze()

            if "silhouette" in renders.keys():
                """
                silhouette = silhouette_renderer(meshes_world=mesh, R=R[num-1:num:], T=T[num-1:num:])
                silhouette = silhouette.cpu().numpy()
                img_dict["silhouette"] = silhouette.squeeze()[...,3]
                """
                # Creating a mask from the image instead of using the silhouette renderer
                silhouette = np.clip(
                    ((img_dict["phong"][..., :3]).astype(np.uint8)) * 255, 0, 255
                )
                silhouette = (silhouette < 1) * 255
                img_dict["silhouette"] = silhouette

            # Merge with background images
            img, background = merge_with_background(
                img_dict["phong"], params, background, show=False
            )
            img_dict["phong"] = img
            if params.show_frame:
                for plot_num, img in enumerate(img_dict.values()):
                    plot_num += 1
                    ax = plt.subplot(1, len(img_dict.values()), plot_num)
                    ax.imshow(img)
                plt.show()

            # Only add the image dict if the image was randomly selected
            if idx - 1 in data_indices:
                # print(num - 1)
                render_manager.add_images(
                    data_img_num, img_dict, R[idx - 1 : idx :], T[idx - 1 : idx :]
                )
                data_img_num += 1

            extra_args = {"compress_level": 3}
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            imageio.imwrite(
                f"tmp/{idx}.png", img_dict["phong"], format="PNG", **extra_args
            )

        image_path_list = []
        t = [s for s in os.listdir("tmp") if s.endswith(".png")]
        for f in sorted(t, key=lambda s: int(re.sub(r"\D", "", s))):
            image_path_list.append(join("tmp", f))

        event_frames = generate_event_frames(
            image_path_list, RenderParams.img_size, RenderParams.batch_size
        )
        event_count = 0
        for ev_count, frame in enumerate(event_frames):
            frame = frame * 4
            all_white = np.zeros((frame.shape), dtype=np.uint8)
            all_white.fill(255)
            frame_black = np.all(frame == [0, 0, 0], axis=-1)
            frame[frame_black] = all_white[frame_black]
            if ev_count in data_indices:
                # print(num)
                render_manager.add_event_frame(event_count, frame)
                event_count += 1

        render_manager.close()
        for f in sorted(os.listdir("tmp")):
            if f.endswith(".png"):
                os.remove(join("tmp", f))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)

    # Set the device
    dev_num = params.gpu_num
    device = torch.device(f"cuda:{dev_num}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} as computation device")
    if device == f"cuda:{dev_num}":
        torch.cuda.set_device()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = dev_num
    logging.info(f"Using {device} as computation device")
    params.device = device

    try:
        main(params)
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating dataset generation")
        sys.exit(0)
