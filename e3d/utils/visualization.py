import os

# Matplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from PIL import Image
# Plotly
from plotly.subplots import make_subplots
from pytorch3d.ops import sample_points_from_meshes



def plot_event_volume(events: np.array, show_grid: bool = True):

    fig = plt.figure(figsize=(25, 12))
    ax = fig.add_subplot(111, projection="3d")
    y = events[:, 0]
    z = events[:, 1]
    x = events[:, 2] * 2  # timestamp
    m = "o"
    c = ["red" if p == 1 else "blue" for p in events[:, 3]]

    ax.scatter3D(x, y, z, c=c, marker=m, s=0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("x [pix]")
    ax.set_zlabel("y [pix]")
    if not show_grid:
        plt.axis("off")
        plt.grid(b=None)
    plt.show()
    # fig.savefig('ev_volume.png', dpi=fig.dpi)


def plot_event_stacks(event_frames: list, show_grid: bool = False):

    z, y = np.ogrid[0 : event_frames[0].shape[0], 0 : event_frames[1].shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x = np.zeros_like(y)
    for i, ev in enumerate(event_frames):
        ev = (
            np.pad(
                ev,
                pad_width=((1, 1), (1, 1), (0, 0)),
                constant_values=0,
                mode="constant",
            )
            / 255
        )
        print(ev.shape)
        ax.plot_surface(
            x + i,
            y,
            z,
            rstride=3,
            cstride=3,
            facecolors=np.rot90(ev, 2, (0, 1)),
            shade=False,
            antialiased=True,
        )
    if not show_grid:
        plt.axis("off")
        plt.grid(b=None)
    plt.show()
    # fig.savefig('stack.png', dpi=fig.dpi)


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    """
    fig = go.Figure()
    fig = px.scatter_3d(
        x = x,
        y = y,
        z = z,
        labels={'x':'x', 'y':'y', 'z':'z'}
    )
    fig.update_layout(
        title_text = f"PointCloud: {title}"
    )
    fig.show()
    """

    # Matplot 3D scatter plot
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


def plot_loss(self, num_losses: int = 1):
    """
    returns a matplotlib subplot object that you can call plot() on
    """
    # plt.axis([-, ,min(self.sequence_score) - .1,1.0])
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.plot([], ".b-", label=self.model.strip(".hdf5"))
    ax.legend(loc="center left", bbox_to_anchor=(0.8, 0.9))
    ax.set_title(self.test_path.split("/")[-1])
    return ax


def plot_camera_scene(cameras, cameras_gt, status: str, device: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
                ground truth locations `cameras_gt`. The plot is named with
                    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    # fig = FigureCanvas(fig)
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, device, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, device, color="#812CE5")
    plot_radius = 10
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    # plt.show()
    return fig


def visualize_vid2e_events(events, resolution):
    """https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_py/tests/plot_virtual_events.py
    """
    pos_events = events[events[:, -1] == 1]
    neg_events = events[events[:, -1] == -1]

    image_pos = np.zeros(resolution[0] * resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0] * resolution[1], dtype="uint8")

    np.add.at(
        image_pos,
        (pos_events[:, 0] + pos_events[:, 1] * resolution[1]).astype("int32"),
        pos_events[:, -1] ** 2,
    )
    np.add.at(
        image_neg,
        (neg_events[:, 0] + neg_events[:, 1] * resolution[1]).astype("int32"),
        neg_events[:, -1] ** 2,
    )

    image_rgb = (
        np.stack(
            [
                image_pos.reshape(resolution),
                image_neg.reshape(resolution),
                np.zeros(resolution, dtype="uint8"),
            ],
            -1,
        )
        * 50
    )

    plt.imshow(image_rgb)
    plt.show()


def plot_img_and_mask(img, mask):

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)

    plt.show()


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(
    ax,
    cameras,
    device: str,
    color: str = "blue",
    width: float = 0.3,
    scale: float = 1.0,
):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale=0.1).to(device)[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    # cam_trans = cameras.get_world_to_view_transform()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float) * 15
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=width)
        plot_handles.append(h)
    return plot_handles


def plot_trajectory_cameras(cameras, device, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.gca(projection="3d")

    ax.clear()
    ax.set_title("Cameras")
    plot_handles = plot_cameras(ax, cameras, device, color="blue", width=0.5)
    # plot_handles_obj = plot_cameras(ax, obj_cam, color="red", width=.7)
    plot_radius = 25
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])

    labels_handles = {"Cameras": plot_handles[0]}
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    # Hide grid lines
    # ax.grid(False)

    # Hide axes ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # plt.axes('off')
    plt.show()
    return fig


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


def to_white_background(path: str):
    # if not os.path.exists(path):
    #    raise Exception("File not found")
    pil_img = Image.open(path)
    img_np = np.array(pil_img)
    all_white = np.zeros((img_np.shape), dtype=np.uint8)
    all_white.fill(255)
    frame_black = np.all(img_np == [0, 0, 0], axis=-1)
    img_np[frame_black] = all_white[frame_black]
    Image.fromarray(img_np).save("test.png")
