import torch
from pytorch3d.ops import sample_points_from_meshes

#Matplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#Plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    '''
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
    '''

    #Matplot 3D scatter plot
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
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


def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
                color `color`.
    """
    cam_wires_canonical = get_camera_wireframe()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
        return plot_handles


def plot_loss(self, num_losses: int = 1):
    """
    returns a matplotlib subplot object that you can call plot() on
    """
    #plt.axis([-, ,min(self.sequence_score) - .1,1.0])
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.plot([], ".b-", label=self.model.strip('.hdf5'))
    ax.legend(loc='center left', bbox_to_anchor=(.8, 0.9))
    ax.set_title(self.test_path.split('/')[-1])
    return ax


def plot_camera_scene(cameras, cameras_gt, status: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
                ground truth locations `cameras_gt`. The plot is named with
                    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
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
    plt.show()
    return fig


def visualize_vid2e_events(events, resolution):
    """https://github.com/uzh-rpg/rpg_vid2e/blob/master/esim_py/tests/plot_virtual_events.py
    """
    pos_events = events[events[:,-1] == 1]
    neg_events = events[events[:,-1] == -1]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    np.add.at(
        image_pos, 
        (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), 
        pos_events[:,-1]**2)
    np.add.at(
        image_neg, 
        (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), 
        neg_events[:,-1]**2)

    image_rgb = np.stack(
        [
            image_pos.reshape(resolution), 
            image_neg.reshape(resolution), 
            np.zeros(resolution, dtype="uint8") 
        ], -1
    ) * 50

    plt.imshow(image_rgb)
    plt.show()

    

