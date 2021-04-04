# Imports and dependencies
import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join("..", dirname(os.getcwd()))))

import json
import logging
import random
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from losses import IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from mesh_reconstruction.params import Params
from mesh_reconstruction.renderer import silhouette_renderer
from pytorch3d.io import load_obj, save_obj
from skimage import img_as_ubyte
from utils.manager import ImageManager, RenderManager


def get_args():
    # TODO Change the argparse error
    parser = argparse.ArgumentParser(
        description="Mesh Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=Params.batch_size,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-mini-b",
        "--mini-batch",
        metavar="B",
        type=int,
        nargs="?",
        default=Params.mini_batch,
        help="Mini batch Size",
        dest="mini_batch",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=Params.learning_rate,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "--img_size",
        dest="img_size",
        type=float,
        default=Params.img_size,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        type=str,
        default=Params.config_file,
        help="Load Params dict from config file, file should be in cfg format",
    )
    parser.add_argument(
        "-name",
        "--experiment_name",
        dest="experiment_name",
        type=str,
        default=Params.experiment_name,
        help="The name of the experiment to run - if ",
    )
    parser.add_argument(
        "-p",
        "--test_path",
        dest="test_path",
        type=str,
        default=Params.test_path,
        help="Path for storing test information",
    )
    parser.add_argument(
        "-show",
        "--show",
        dest="show",
        type=bool,
        default=Params.test_path,
        help="Boolean indicator on whether to plot results or not",
    )

    return parser.parse_args()


def plot_and_save(elems: list, path: str, name: str):
    plt.plot(elems, label=f"{name} Loss")
    plt.legend(fontsize="16")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs iterations")
    plt.savefig(join(path, f"{name}_loss.png"))
    plt.close()


def sample_tensor(t, batch_size: int = 0, indices=None):
    l = t.shape[0]
    if batch_size and l < batch_size:
        return
    if not indices:
        step = int(l / batch_size)
        start = random.randint(0, step - 1)
        indices = list(range(start, l, step))
    return t[indices], indices


def generate_indices():

    # Create path to tests folder
    tests_path = "../data/tests/dolphin-sep23-baseline"
    batch_size = int(360 / 8)  # =45
    ratios = [0.25, 0.50, 0.75, 0.100]
    # Fetch indices
    with open(join(tests_path, "indices.json"), "r") as f:
        indices = json.load(f)
    # Generate all the good indices and create sets of experiments
    experiments = {}
    good_indices = set(range(360)) - set(
        [elem for ind in indices.values() for elem in ind]
    )
    for name, ind in indices.items():
        if name == "outliers":
            continue
        for ratio in ratios:
            num_clean = int((1 - ratio) * batch_size)
            num_noisy = batch_size - num_clean

            ind_clean = sorted(random.choices(list(good_indices), k=num_clean))
            ind_noisy = sorted(random.choices(ind, k=num_noisy))

            experiment_indices = ind_clean + ind_noisy

            experiments[
                f"{name}_{int(100*ratio)}-{100-int(100*ratio)}"
            ] = experiment_indices

    experiments = {}
    experiments["best_1"] = sorted(random.choices(list(good_indices), k=batch_size))
    experiments["best_2"] = sorted(random.choices(list(good_indices), k=batch_size))
    experiments["best_3"] = sorted(random.choices(list(good_indices), k=batch_size))


'''
def test(params: Params):
    """
    Completes a test run and saves all the relevant information
    """

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

    results = {}
    results["indices"] = indices

    # We will save images periodically and compose them into a GIF.
    filename_output = join(path, "projection_loss.gif")
    writer = imageio.get_writer(filename_output, mode="I", duration=0.1)

    result = run()

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
'''

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.cuda.set_device()

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)
    params.device = device

    try:
        # test(params)
        raise NotImplementedError
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating test")
        sys.exit(0)
