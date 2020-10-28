import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from losses import IOULoss
from mesh_reconstruction.params import Params
from mesh_reconstruction.renderer import silhouette_renderer
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import SfMPerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from skimage import img_as_ubyte
from torchvision import transforms
from tqdm import tqdm_notebook

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MeshDeformationModel(nn.Module):
    def __init__(self, device, params=Params(), template_mesh=None):
        super().__init__()

        self.device = device
        self.params = params

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

        # Create optimizer
        self.optimizer = self.params.optimizer(
            self.parameters(), lr=self.params.learning_rate, betas=self.params.betas
        )

        self.losses = {"iou": [], "laplacian": [], "flatten": []}

        # Create a silhouette_renderer
        self.renderer = silhouette_renderer(self.params.img_size, device)

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

    def run_optimization(
        self,
        silhouettes: torch.tensor,
        R: torch.tensor,
        T: torch.tensor,
        writer=None,
        step: int = 0,
    ):
        """
        Function:
            Runs a batched optimization procedure that aims to minimize 3 reconstruction losses:
                -Silhouette IoU Loss: between input silhouettes and re-projected mesh
                -Mesh Edge consistency
                -Mesh Normal smoothing
            Mini Batching:
                If the number silhouettes is greater than the allowed batch size then a random set of images/poses is sampled for supervision at each step
        """

        if len(R.shape) == 4:
            R = R.squeeze(1)
            T = T.squeeze(1)

        tf_smaller = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.params.img_size),
                transforms.ToTensor(),
            ]
        )
        tf_original = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(silhouettes.shape[-1]),
                transforms.ToTensor(),
            ]
        )
        images_gt = torch.stack(
            [tf_smaller(s.cpu()).to(self.device) for s in silhouettes]
        ).squeeze(1)
        if images_gt.max() > 1.0:
            images_gt = images_gt / 255.0

        loop = tqdm_notebook(range(self.params.steps))

        start_time = time.time()
        for i in loop:

            batch_indices = (
                random.choices(
                    list(range(images_gt.shape[0])), k=self.params.batch_size
                )
                if images_gt.shape[0] > self.params.batch_size
                else list(range(images_gt.shape[0]))
            )
            batch_silhouettes = images_gt[batch_indices]

            batch_R, batch_T = R[batch_indices], T[batch_indices]
            batch_cameras = SfMPerspectiveCameras(
                device=self.device, R=batch_R, T=batch_T
            )

            mesh, laplacian_loss, flatten_loss = self.forward(self.params.batch_size)

            images_pred = self.renderer(
                mesh.clone(), device=self.device, cameras=batch_cameras
            )[..., -1]

            iou_loss = IOULoss().forward(batch_silhouettes, images_pred)
            # ssd_loss = torch.sum((images_gt - images_pred[...,-1]) ** 2).mean()

            loss = (
                iou_loss * self.params.lambda_iou
                + laplacian_loss * self.params.lambda_laplacian
                + flatten_loss * self.params.lambda_flatten
            )

            loop.set_description("Optimizing (loss %.4f)" % loss.data)

            self.losses["iou"].append(iou_loss * self.params.lambda_iou)
            self.losses["laplacian"].append(
                laplacian_loss * self.params.lambda_laplacian
            )
            self.losses["flatten"].append(flatten_loss * self.params.lambda_flatten)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0 and self.params.show:
                # Write images
                image = images_pred.detach().cpu().numpy()[0][..., -1]

                if writer:
                    writer.append_data((255 * image).astype(np.uint8))
                # imageio.imsave(join(path, f"mesh_{i}.png"), (255*image).astype(np.uint8))

                if self.params.show:
                    f, (ax1, ax2) = plt.subplots(1, 2)

                    image = img_as_ubyte(image)

                    ax1.imshow(image)
                    ax1.set_title("Deformed Mesh")

                    # ax2.plot(silhouette_losses, label="Silhouette Loss")
                    # ax2.plot(laplacian_losses, label="Laplacian Loss")
                    # ax2.plot(flatten_losses, label="Flatten Loss")
                    ax2.legend(fontsize="16")
                    ax2.set_xlabel("Iteration", fontsize="16")
                    ax2.set_ylabel("Loss", fontsize="16")
                    ax2.set_title("Loss vs iterations", fontsize="16")

                    plt.show()

        # Create a silhouette projection of the mesh across all views
        all_silhouettes = []
        for i in range(0, images_gt.shape[0]):
            t_cameras = SfMPerspectiveCameras(device=self.device, R=R[[i]], T=T[[i]])
            all_silhouettes.append(
                tf_original(
                    self.renderer(
                        mesh[0].clone(), device=self.device, cameras=t_cameras
                    )
                    .detach()
                    .cpu()[..., -1]
                )
            )
        all_silhouettes = torch.cat(all_silhouettes).unsqueeze(-1).permute(0, 3, 1, 2)

        # mean_iou = IOULoss().forward(images_gt.detach().cpu(), all_silhouettes[...,-1].detach().cpu()).detach().cpu().numpy().tolist()

        results = dict(
            silhouette_loss=[
                s.detach().cpu().numpy().tolist() for s in self.losses["iou"]
            ],
            laplacian_loss=[
                s.detach().cpu().numpy().tolist() for s in self.losses["laplacian"]
            ],
            flatten_loss=[
                s.detach().cpu().numpy().tolist() for s in self.losses["flatten"]
            ],
            iterations_per_second=self.params.steps / (time.time() - start_time),
        )

        return (all_silhouettes, results)
