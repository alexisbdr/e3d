import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os


def plot_event_volume(events: np.array, path: str, scale: float=0.2,  show_grid: bool = True, N: int = 10):
    fig = plt.figure(figsize=(25, 12), dpi=288)
    ax = fig.add_subplot(111, projection="3d")
    y = events[:, 1]
    z = events[:, 2]
    x = events[:, 0]  # timestamp
    m = "o"
    c = ["red" if p == 1 else "blue" for p in events[:, 3]]
    ax.scatter3D(x, y, z, c=c, marker=m, s=scale)
    ax.set_xlabel("Time [s]", fontsize=N+4, fontweight='bold', labelpad=18)
    ax.set_ylabel("x [pix]", fontsize=N+4, fontweight='bold', labelpad=18)
    ax.set_zlabel("y [pix]", fontsize=N+4, fontweight='bold', labelpad=18)
    ax.tick_params(axis='both', which='major', labelsize=N)
    ax.tick_params(axis='both', which='minor', labelsize=N)

    if not show_grid:
        plt.axis("off")
        plt.grid(b=None)
    # plt.show()
    fig.savefig(path, dpi=fig.dpi)

event_path = "/data4/jiangjianping/E3D/EVIMO/data_pre/plane_small/seq_02/events.txt"
# event_path = "/data2/jiangjianping/Event/test.txt"
dir_path = "./data/train_second/test_paper/floor_seq_01/volumes"
#dir_path = "./data/plane_train_third/test_paper/seq_02/"
os.makedirs(dir_path, exist_ok=True)
scale = 0.2

if __name__ == '__main__':
    event = np.loadtxt(event_path)
    nums = [30000]
    for item in nums:
        if event.shape[0] > item:
            skip = int(event.shape[0] / item)
            event = event[:5*item]
            event = event[::5]
            t_scale = 60. / event[-1, 0]
            event[:, 0] = event[:, 0] * t_scale
        for size in [14, 16, 18]:
            path = os.path.join(dir_path, f'{size}.png')
            plot_event_volume(event, path, scale, N=size)

