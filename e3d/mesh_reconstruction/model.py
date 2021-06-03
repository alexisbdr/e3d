import logging
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from losses import IOULoss
from utils.params import Params
from mesh_reconstruction.renderer import flat_renderer, silhouette_renderer
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from torchvision import transforms
from tqdm import tqdm_notebook
from pytorch3d.transforms import Transform3d, matrix_to_quaternion, quaternion_to_matrix
from utils.pyutils import _broadcast_bmm
from utils.visualization import plot_pointcloud

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MeshDeformationModel(nn.Module):
    def __init__(self, device, params=Params(), template_mesh=None):
        super().__init__()

        self.device = device
        self.params = params
        self.mesh_scale = params.mesh_sphere_scale
        self.ico_level = params.mesh_sphere_level
        self.is_real_data = params.is_real_data
        self.init_pose_R = None
        self.init_pose_t = None

        # Create a source mesh
        if not template_mesh:
            template_mesh = ico_sphere(params.mesh_sphere_level, device)
            template_mesh.scale_verts_(params.mesh_sphere_scale)

        # for EVIMO data, we need to apply a delta Transform to adjust the pose in the EVIMO coordinate system
        # to PyTorch3D system
        # Since we don't know the initial transform, we optimize the initial pose as a parameter while render the mesh
        # initialize the delta Transform
        if params.is_real_data:
            init_trans = Transform3d(device=device)
            R_init = init_trans.get_matrix()[:, :3, :3]
            qua_init = matrix_to_quaternion(R_init)
            random_noise = (torch.randn(qua_init.shape) / params.mesh_pose_init_noise_var).to(self.device)
            qua_init += random_noise

            t_init = init_trans.get_matrix()[:, 3:, :3]
            random_noise_t = (torch.randn(t_init.shape) / params.mesh_pose_init_noise_var).to(self.device)
            t_init += random_noise_t

            self.register_parameter(
                'init_camera_R', nn.Parameter(qua_init).to(self.device)
            )
            self.register_parameter(
                'init_camera_t', nn.Parameter(t_init).to(self.device)
            )

        verts, faces = template_mesh.get_mesh_verts_faces(0)
        # Initialize each vert to have no tetxture
        verts_rgb = torch.ones_like(verts)[None]
        textures = TexturesVertex(verts_rgb.to(self.device))
        self.template_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures,
        )

        self.register_buffer("vertices", self.template_mesh.verts_padded())
        self.register_buffer("faces", self.template_mesh.faces_padded())
        self.register_buffer("textures", textures.verts_features_padded())

        deform_verts = torch.zeros_like(
            self.template_mesh.verts_packed(), device=device, requires_grad=True
        )
        # Create an optimizable parameter for the mesh
        self.register_parameter(
            "deform_verts", nn.Parameter(deform_verts).to(self.device)
        )

        # Create optimizer
        self.optimizer = self.params.mesh_optimizer(
            self.parameters(), lr=self.params.mesh_learning_rate, betas=self.params.mesh_betas
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

    @property
    def _final_mesh(self):
        """Protected getter for the final optimized mesh
        """
        assert (
            "final_mesh" in self.__dict__.keys()
        ), "Final Mesh does not exist yet - please run multi-view optimization before getting"
        return self.final_mesh

    @property
    def _get_init_pose(self):
        """
        Get the initial pose of the EVIMO mesh model
        """
        return self.init_pose_R, self.init_pose_t

    def render_final_mesh(self, poses, mode: str, out_size: list, camera_settings=None) -> dict:
        """Renders the final mesh obtained through optimization
            Supports two modes:
                -predict: renders both silhouettes and flat shaded images
                -train: only renders silhouettes
            Returns:
                -dict of renders {'silhouettes': tensor, 'images': tensor}
        """
        R, T = poses
        if len(R.shape) == 4:
            R = R.squeeze(1)
            T = T.squeeze(1)

        sil_renderer = silhouette_renderer(out_size, self.device)
        image_renderer = flat_renderer(out_size, self.device)

        # Create a silhouette projection of the mesh across all views
        all_silhouettes = []
        all_images = []
        for i in range(0, R.shape[0]):
            batch_R, batch_T = R[[i]], T[[i]]
            if self.params.is_real_data:
                init_R = quaternion_to_matrix(self.init_camera_R)
                batch_R = _broadcast_bmm(batch_R, init_R)
                batch_T = (_broadcast_bmm(batch_T[:, None, :], init_R)
                           + self.init_camera_t.expand(batch_R.shape[0], 1, 3))[:, 0, :]
                focal_length = torch.tensor([camera_settings[0, 0], camera_settings[1, 1]])[None]
                principle_point = torch.tensor([camera_settings[0, 2], camera_settings[1, 2]])[None]
                t_cameras = PerspectiveCameras(device=self.device, R=batch_R, T=batch_T,
                                               focal_length=focal_length,
                                               principal_point=principle_point,
                                               image_size=((self.params.img_size[1], self.params.img_size[0]),)
                                               )
            else:
                t_cameras = PerspectiveCameras(device=self.device, R=batch_R, T=batch_T)
            all_silhouettes.append(
                sil_renderer(self._final_mesh, device=self.device, cameras=t_cameras)
                .detach()
                .cpu()[..., -1]
            )

            if mode == "predict":
                all_images.append(
                    torch.clamp(
                        image_renderer(
                            self._final_mesh, device=self.device, cameras=t_cameras
                        ),
                        0,
                        1,
                    )
                    .detach()
                    .cpu()[..., :3]
                )
            torch.cuda.empty_cache()
        renders = dict(
            silhouettes=torch.cat(all_silhouettes).unsqueeze(-1).permute(0, 3, 1, 2),
            images=torch.cat(all_images) if all_images else [],
        )

        return renders

    def run_optimization(
        self,
        silhouettes: torch.tensor,
        R: torch.tensor,
        T: torch.tensor,
        writer=None,
        camera_settings=None,
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
        Returns:
            -Reconstruction losses: 3 reconstruction losses measured during optimization
            -Timing:
                -Iterations / second
                -Total time elapsed in seconds
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

        images_gt = torch.stack(
            [tf_smaller(s.cpu()).to(self.device) for s in silhouettes]
        ).squeeze(1)

        if images_gt.max() > 1.0:
            images_gt = images_gt / 255.0

        loop = tqdm_notebook(range(self.params.mesh_steps))

        start_time = time.time()
        for i in loop:
            batch_indices = (
                random.choices(
                    list(range(images_gt.shape[0])), k=self.params.mesh_batch_size
                )
                if images_gt.shape[0] > self.params.mesh_batch_size
                else list(range(images_gt.shape[0]))
            )
            batch_silhouettes = images_gt[batch_indices]

            batch_R, batch_T = R[batch_indices], T[batch_indices]
            # apply right transform on the Twv to adjust the coordinate system shift from EVIMO to PyTorch3D
            if self.params.is_real_data:
                init_R = quaternion_to_matrix(self.init_camera_R)
                batch_R = _broadcast_bmm(batch_R, init_R)
                batch_T = (_broadcast_bmm(batch_T[:, None, :], init_R) +
                           self.init_camera_t.expand(batch_R.shape[0], 1, 3))[:, 0, :]
                focal_length = (torch.tensor([camera_settings[0, 0], camera_settings[1, 1]])[None]).expand(
                    batch_R.shape[0], 2)
                principle_point = (torch.tensor([camera_settings[0, 2], camera_settings[1, 2]])[None]).expand(
                    batch_R.shape[0], 2)
                # FIXME: in this PyTorch3D version, the image_size in RasterizationSettings is (W, H), while in PerspectiveCameras is (H, W)
                # If the future pytorch3d change the format, please change the settings here
                # We hope PyTorch3D will solve this issue in the future
                batch_cameras = PerspectiveCameras(
                    device=self.device, R=batch_R, T=batch_T, focal_length=focal_length,
                    principal_point=principle_point,
                    image_size=((self.params.img_size[1], self.params.img_size[0]),)
                )
            else:
                batch_cameras = PerspectiveCameras(
                    device=self.device, R=batch_R, T=batch_T
                )

            mesh, laplacian_loss, flatten_loss = self.forward(self.params.mesh_batch_size)

            images_pred = self.renderer(
                mesh, device=self.device, cameras=batch_cameras
            )[..., -1]

            iou_loss = IOULoss().forward(batch_silhouettes, images_pred)

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
            if i % (self.params.mesh_show_step / 2) == 0 and self.params.mesh_log:
                logging.info(f'Iteration: {i} IOU Loss: {iou_loss.item()} Flatten Loss: {flatten_loss.item()} Laplacian Loss: {laplacian_loss.item()}')

            if i % self.params.mesh_show_step == 0 and self.params.im_show:
                # Write images
                image = images_pred.detach().cpu().numpy()[0]

                if writer:
                    writer.append_data((255 * image).astype(np.uint8))
                plt.imshow(images_pred.detach().cpu().numpy()[0])
                plt.show()
                plt.imshow(batch_silhouettes.detach().cpu().numpy()[0])
                plt.show()
                plot_pointcloud(mesh[0], 'Mesh deformed')
                logging.info(f'Pose of init camera: {self.init_camera_R.detach().cpu().numpy()}, {self.init_camera_t.detach().cpu().numpy()}')


        # Set the final optimized mesh as an internal variable
        self.final_mesh = mesh[0].clone()
        results = dict(
            silhouette_loss=self.losses["iou"][-1].detach().cpu().numpy().tolist(),
            laplacian_loss=self.losses["laplacian"][-1].detach().cpu().numpy().tolist(),
            flatten_loss=self.losses["flatten"][-1].detach().cpu().numpy().tolist(),
            iterations_per_second=self.params.mesh_steps / (time.time() - start_time),
            total_time_s=time.time() - start_time,
        )
        if self.is_real_data:
            self.init_pose_R = self.init_camera_R.detach().cpu().numpy()
            self.init_pose_t = self.init_camera_t.detach().cpu().numpy()

        torch.cuda.empty_cache()

        return results
