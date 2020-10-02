# Imports and dependencies
import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join("..", dirname(os.getcwd()))))

import json
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import (BlendParams, HardFlatShader, HardPhongShader,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SfMPerspectiveCameras,
                                SoftSilhouetteShader, TexturesVertex,
                                look_at_rotation, look_at_view_transform)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.utils import ico_sphere
from skimage import img_as_ubyte
from tqdm import tqdm_notebook
from utils.manager import ImageManager, RenderManager
from utils.shapes import Sphere, SphericalSpiral
from utils.visualization import plot_pointcloud


def plot_and_save(elems: list, path: str, name: str):
    plt.plot(elems, label=f"{name} Loss")
    plt.legend(fontsize="16")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs iterations")
    plt.savefig(join(path, f"{name}_loss.png"))
    plt.close()


def sample_tensor(t, batch_size, indices=None):
    l = t.shape[0]
    if l < batch_size:
        return
    if not indices:
        step = int(l / batch_size)
        start = random.randint(0, step - 1)
        indices = list(range(start, l, step))
    return t[indices], indices


class MeshDeformationModel(nn.Module):
    def __init__(self, device, template_mesh=None):
        super().__init__()

        self.device = device

        # Create a source mesh
        if not template_mesh:
            template_mesh = ico_sphere(2, device)

        verts, faces = template_mesh.get_mesh_verts_faces(0)
        # Initialize each vert to have no tetxture
        verts_rgb = torch.ones_like(verts)[None]
        textures = TexturesVertex(verts_rgb.to(self.device))
        self.template_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures,
        )

        self.register_buffer("vertices", self.template_mesh.verts_padded() * 1.3)
        self.register_buffer("faces", self.template_mesh.faces_padded())
        self.register_buffer("textures", textures.verts_features_padded())

        deform_verts = torch.zeros_like(
            self.template_mesh.verts_packed(), device=device, requires_grad=True
        )
        # deform_verts = torch.full(self.template_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
        # Create an optimizable parameter for the mesh
        self.register_parameter(
            "deform_verts", nn.Parameter(deform_verts).to(self.device)
        )

        laplacian_loss = mesh_laplacian_smoothing(template_mesh, method="uniform")
        flatten_loss = mesh_normal_consistency(template_mesh)

    def forward(self, batch_size):
        # Offset the mesh
        deformed_mesh_verts = self.template_mesh.offset_verts(self.deform_verts)
        texture = TexturesVertex(self.textures)
        deformed_mesh = Meshes(
            verts=deformed_mesh_verts.verts_padded(),
            faces=deformed_mesh_verts.faces_padded(),
            textures=texture,
        )
        deformed_meshes = deformed_mesh.extend(batch_size)

        laplacian_loss = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
        flatten_loss = mesh_normal_consistency(deformed_mesh)

        return deformed_meshes, laplacian_loss, flatten_loss


def run():

    weight_silhouette = 1
    weight_laplacian = 0.1
    weight_flatten = 0.001

    batch_size = int(360 / 6)  # =60

    # Create a loss plotting object
    # loss_ax = plot_loss(num_losses = 3)

    # Initialize a model using the renderer, template mesh and reference image
    model = MeshDeformationModel(device).to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.5, 0.99)
    )  # Hyperparameter tuning

    # Create path to tests folder
    experiment_name = f"random_indices_{i+1}"
    tests_path = "../data/tests/dolphin-sep23-baseline"
    path = join(tests_path, experiment_name)
    os.makedirs(path, exist_ok=True)

    # Fetch indices
    with open(join(path, "indices.json"), "r") as f:
        indices = json.load(f)
    all_indices = set(range(360))

    # render = RenderManager.from_directory(dir_num=60)
    render = RenderManager.from_path(
        "../data/renders/test_dolphin/002-dolphin_2020-09-23T11:45:50/"
    )
    R, T = render._trajectory
    indices = None
    R, indices = sample_tensor(R, batch_size)
    T, _ = sample_tensor(T, batch_size, indices)

    images_gt = render._images(type_key="silhouette_pred", img_size=img_size).to(device)
    images_gt, _ = sample_tensor(images_gt, batch_size, indices)

    cameras = SfMPerspectiveCameras(device=device, R=R, T=T)

    results = {}
    results["indices"] = indices

    # We will save images periodically and compose them into a GIF.
    filename_output = join(path, "projection_loss.gif")
    writer = imageio.get_writer(filename_output, mode="I", duration=0.1)

    loop = tqdm_notebook(range(2000))
    laplacian_losses = []
    flatten_losses = []
    silhouette_losses = []

    for i in loop:

        mesh, laplacian_loss, flatten_loss = model(batch_size)

        images_pred = silhouette_renderer(mesh.clone(), device=device, cameras=cameras)

        silhouette_loss = neg_iou_loss(images_gt, images_pred[..., -1])
        # ssd_loss = torch.sum((images_gt - images_pred[...,-1]) ** 2).mean()

        loss = (
            silhouette_loss * weight_silhouette
            + laplacian_loss * weight_laplacian
            + flatten_loss * weight_flatten
        )

        loop.set_description("Optimizing (loss %.4f)" % loss.data)

        silhouette_losses.append(silhouette_loss * weight_silhouette)
        laplacian_losses.append(laplacian_loss * weight_laplacian)
        flatten_losses.append(flatten_loss * weight_flatten)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            # Write images
            image = images_pred.detach().cpu().numpy()[0][..., -1]

            writer.append_data((255 * image).astype(np.uint8))
            # imageio.imsave(join(path, f"mesh_{i}.png"), (255*image).astype(np.uint8))

            f, (ax1, ax2) = plt.subplots(1, 2)

            image = img_as_ubyte(image)
            ax1.imshow(image)
            ax1.set_title("Deformed Mesh")

            ax2.plot(silhouette_losses, label="Silhouette Loss")
            ax2.plot(laplacian_losses, label="Laplacian Loss")
            ax2.plot(flatten_losses, label="Flatten Loss")
            ax2.legend(fontsize="16")
            ax2.set_xlabel("Iteration", fontsize="16")
            ax2.set_ylabel("Loss", fontsize="16")
            ax2.set_title("Loss vs iterations", fontsize="16")

            plt.show()

        # Save obj, gif, individual losses, mesh similarity metric
        verts, faces = mesh.get_mesh_verts_faces(0)
        save_obj(join(path, "mesh.obj"), verts, faces)

    plot_and_save(silhouette_losses, "silhouette")
    plot_and_save(laplacian_losses, "laplacian")
    plot_and_save(flatten_losses, "flatten")

    results["silhouette_loss"] = [
        s.detach().cpu().numpy().tolist() for s in silhouette_losses
    ]
    results["laplacian_loss"] = [
        l.detach().cpu().numpy().tolist() for l in laplacian_losses
    ]
    results["flatten_loss"] = [
        f.detach().cpu().numpy().tolist() for f in flatten_losses
    ]
    with open(join(path, "results.json"), "w") as f:
        json.dump(results, f)

    writer.close()


if __name__ == "__main__":

    # Matplotlib config nums
    mpl.rcParams["savefig.dpi"] = 90
    mpl.rcParams["figure.dpi"] = 90
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.cuda.set_device()

    img_size = (64, 64)

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
        image_size=img_size[0],
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
        image_size=img_size[0], blur_radius=1e-5, faces_per_pixel=1,
    )
    # We can add a point light in front of the object.
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
