"""
#imports and dependencies
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join("..", dirname(os.getcwd()))))

import random
from dataclasses import asdict, astuple, dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from synth_dataset.trajectory import cam_trajectory
from tqdm import tqdm_notebook
from utils.manager import ImageManager, RenderManager
from utils.visualization import plot_trajectory_cameras


@datataclass
class RenderParams:

    img_size: int = (560, 560)
    sigma_hand: float = 0.15

    # Size of the dataset
    mini_batch: int = 72
    batch_size: int = 360
    mesh_iter: int = 4

    show_frame: bool = False


cameras = SfMPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
# edges. Refer to blending.py for more details.
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=RenderParams.img_size[0],
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
    faces_per_pixel=100,
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader.
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=blend_params),
)


# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=RenderParams.img_size[0], blur_radius=1e-5, faces_per_pixel=100,
)
# We can add a point light in front of the object.
# lights = PointLights(device=device, location=((2., 2.0, 2.0),))
lights = PointLights(
    device=device,
    location=[[3.0, 3.0, 0.0]],
    diffuse_color=((1.0, 1.0, 1.0),),
    specular_color=((1.0, 1.0, 1.0),),
)

phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardFlatShader(device=device, lights=lights, cameras=cameras),
)

# Load obj file
path = "../data/meshes/dolphin/dolphin.obj"
mesh = load_objs_as_meshes(
    [path], create_texture_atlas=False, load_textures=True, device=device
)
# mesh.textures = TexturesVertex(
#                verts_features=torch.ones_like(mesh.verts_padded(), device=device)
#            )

verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center.expand(N, 3))
mesh.scale_verts_((1.0 / float(scale)))


mesh = rotate_mesh_around_axis(
    mesh, [90, 90, 180], phong_renderer, dist=2, device=device
)
# mesh = translate_mesh_on_axis(mesh, [0,-20,-50], phong_renderer, dist=5)
# verts, faces = mesh.get_mesh_verts_faces(0)

# save_obj("../data/meshes/plane_WWII/plane_WWII.obj", verts, faces)

"""
