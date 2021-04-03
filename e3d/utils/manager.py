import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join("..", dirname(__file__))))

import json
import logging
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import List

import imageio
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s, %(message)s")

from utils.pyutils import _from_dict


def default_tensor():
    return torch.tensor([])


@dataclass
class    EventFrameManager:
    """
    Contains information about a single event frame
    Serializes/De-Serialized the event frame using pickle
    """

    posn: int = 0

    file_name: str = ""
    extension: str = "npy"

    @property
    def _dict(self):
        return asdict(self)

    @property
    def _load(self):
        if not os.path.exists(self.file_name):
            raise Exception(f"Path {self.file_name} for event frame does not exist")
        if self.extension == "npy":
            return np.load(self.file_name)
        elif self.extension == "png":
            return Image.open(self.file_name).convert("L")

    def _save(self, event_data, f_loc, sformat: str = ""):
        self.extension = sformat if sformat else self.extension
        if type(event_data) is not np.ndarray:
            raise Exception("Event Data should be a Numpy Array")
        event_file = f"{self.posn}_event.{self.extension}"
        self.file_name = join(f_loc, event_file)
        if self.extension == "npy":
            np.save(self.file_name, event_data)
        elif self.extension == "png":
            img_format = self.extension.upper()
            extra_args = {"compress_level": 3}
            imageio.imwrite(self.file_name, event_data, format=img_format, **extra_args)

    @classmethod
    def from_dict(cls, dict_in):
        return _from_dict(cls, dict_in)


@dataclass
class ImageManager:
    """
    Contains information about a single rendered image
    Serializes/De-Serializes the image using imageio test
    """

    # [required]: position in the render
    posn: int = 0

    image_path: str = ""

    # one of "silhouette", "shaded" or "textured"
    render_type: str = ""

    # Camera pose at that render point
    R: list = field(default_factory=list)
    T: list = field(default_factory=list)

    extension: str = "jpg"

    @property
    def _dict(self):
        return asdict(self)

    @property
    def _load(self) -> Image:
        if not os.path.exists(self.image_path):
            raise Exception(f"Image path {self.image_path} does not exist")
        return Image.open(self.image_path).convert("L")

    def gray(self, img):
        if isinstance(img, np.array):
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img.convert("L")

    def _save(self, image_data, f_loc, img_format="png"):
        if img_format == "jpg":
            self.extension = img_format
            img_format = "JPEG-PIL"
            extra_args = {}
        elif img_format == "png":
            self.extension = img_format
            img_format = "PNG"
            # img_format = img_format.upper()
            # Lossy conversion from float32 to uint8
            """
            info = np.finfo(image_data.dtype)
            #Normalize the image
            image_data /= info.max
            image_data *= 255
            image_data = image_data.astype(np.uint8)
            """
            # Lowered the compression level for improved performance
            # Refer to this issue https://github.com/imageio/imageio/issues/387
            extra_args = {"compress_level": 3}
        elif img_format == "tif":
            self.extension = img_format
            img_format = img_format.upper()
            extra_args = {}
        img_file = f"{self.posn}_{self.render_type}.{self.extension}"
        self.image_path = join(f_loc, img_file)
        imageio.imwrite(self.image_path, image_data, format=img_format, **extra_args)

    @classmethod
    def from_dict(cls, dict_in):
        return _from_dict(cls, dict_in)


@dataclass
class RenderManager:
    """
    Manages incoming rendered images and places them into catalog folder (numbered 0 - n)
    Creates a json file with meta information about the folder and details about the images
    Creates a gif of the render
    """

    mesh_name: str = ""
    # List of paths on disk - storing the dataclass here might make it too large (to test)
    images: dict = field(default_factory=dict)

    # Trajectory
    R: list = field(default_factory=list)
    T: list = field(default_factory=list)

    # List of render types
    types: list = field(default_factory=list)
    # Stored data
    metadata: dict = field(default_factory=dict)
    pred_results: dict = field(default_factory=dict)

    # Internally managed
    count: int = 0  # This is a count of poses not total images
    folder_locs: dict = field(default_factory=dict)
    formatted_utc_ts: str = ""
    gif_writers: dict = field(default_factory=dict)

    render_folder: str = "data/renders/"
    new_folder: str = ""
    folder_name: str = ""

    def init(self):
        """Initialization step for the manager
            1. creates folder architecture on disk
            2. initializes gif writers and image dict
        """
        # Timestamp format
        curr_struct_UTC_ts = time.gmtime(time.time())
        self.formatted_utc_ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Construct absolute
        curr_path = __file__
        path_to_e3d = dirname(dirname(curr_path))
        self.base_folder = join(path_to_e3d, self.render_folder, self.new_folder)
        os.makedirs(self.base_folder, exist_ok=True)

        nums = [0]
        for f in os.listdir(self.base_folder):
            try:
                nums.append(int(f.split("-")[0]))
            except:
                continue
        render_posn = max(nums) + 1
        if not self.folder_name:
            self.folder_name = (
                f"{render_posn:03}-{self.mesh_name}_{self.formatted_utc_ts}"
            )
        self.folder_locs["base"] = os.path.join(self.base_folder, self.folder_name,)
        logging.info(f"Render Manager started in base file {self.folder_locs['base']}")
        for t in self.types:
            if t not in self.allowed_render_types:
                raise TypeError(
                    f"RenderManager: Wrong image type set in init, an image type must be one of: {self.allowed_render_types}"
                )
            # Create a folder for each type
            self.folder_locs[t] = join(self.folder_locs["base"], t)
            os.makedirs(self.folder_locs[t], exist_ok=True)
            self.open_gif_writer(t)
            self.images[t] = []

    def get_trajectory_point(self, index: int) -> tuple:
        assert (
            index >= 0 and index < self.count
        ), "Index out of bounds to Trajectory Points"
        R = torch.tensor(self.R[index])
        T = torch.tensor(self.T[index])
        return (R, T)

    @property
    def _trajectory(self) -> tuple:
        R = torch.stack(([torch.tensor(r) for r in self.R]))[:, 0, :]
        T = torch.stack(([torch.tensor(t) for t in self.T]))[:, 0, :]
        return (R, T)

    @property
    def allowed_render_types(self):
        return [
            "silhouette",
            "phong",
            "textured",
            "events",
            "silhouette_pred",
            "mesh_silhouette",
            "mesh_textured",
        ]

    def open_gif_writer(self, t: str, duration: float = 0.1):

        if t in self.gif_writers:
            return
        gif_t_loc = join(self.folder_locs[t], f"camera_simulation_{t}.gif")
        gif_t_writer = imageio.get_writer(gif_t_loc, mode="I", duration=duration)
        self.gif_writers[t] = gif_t_writer

    def __len__(self):
        return self.count

    def get_image(self, type_key: str, index: int, img_size: tuple = (0, 0)) -> Image:
        """Returns a torch tensor of the loaded image at the given index
        """
        assert (
            index >= 0 and index <= self.count
        ), f"Index {index} out of bounds for image type {type_key}"
        assert type_key in self.images, f"Incorrect Type {type_key}"
        img_dict = deepcopy(self.images[type_key][index])
        img_manager = ImageManager.from_dict(img_dict)
        image = img_manager._load
        if img_size[0]:
            image = image.resize(img_size)
        return image

    def _images(self, type_key: str = "phong", img_size: tuple = (0, 0)) -> list:
        """Returns a stacked tensor of image tensors
        """
        images_data = [
            torch.from_numpy(np.array(self.get_image(type_key, num, img_size)))
            for num in range(self.count)
        ]
        return torch.stack(images_data)

    def get_event_frame(self, index: int, img_size: tuple = (0, 0)) -> Image:
        """Returns a torch tensor of the loaded event frame at the given index
        """
        assert (
            index >= 0 and index <= self.count
        ), f"Index {index} out of bounds for events"
        event_dict = deepcopy(self.images["events"][index])
        event_manager = EventFrameManager.from_dict(event_dict)
        event_frame = event_manager._load
        if img_size[0]:
            event_frame = event_frame.resize(img_size)
        return event_frame

    def _events(self, img_size: tuple = (0, 0)) -> list:
        """Returns torch tensor of event frames for the given key
        """
        assert "events" in self.images, "No events available for this Render Manager"
        event_data = [
            torch.from_numpy(np.array(self.get_event_frame(num)))
            for num in range(len(self.images["events"]))
        ]

        return torch.stack(event_data)

    def add_images(self, count, imgs_data, R, T):
        # Create ImageData class for each type of image
        R = R.tolist()
        T = T.tolist()
        for img_type in imgs_data.keys():
            if img_type not in self.images.keys():
                raise TypeError(f"RenderManager: wrong render type {img_type}")
            img_manager = ImageManager(posn=count, render_type=img_type, R=R, T=T)
            img_manager._save(imgs_data[img_type], self.folder_locs[img_type])
            # Append to gif writer
            img = img_as_ubyte(imgs_data[img_type])
            self.gif_writers[img_type].append_data(img)
            # Append to images list
            self.images[img_type].append(img_manager._dict)
        self.count += 1
        if not len(R) and not len(T):
            self.R = R
            self.T = T
        else:
            self.R.append(R)
            self.T.append(T)

    def add_pred(self, num, pred, type_key: str, destination: str = ""):
        """Adds predictions to the manager
            Predictions can be either: mask predictions, predicted renders (supported for the time being)
            num goes from 0 - n
        """
        # First create a folder for the pred
        assert (
            type_key in self.allowed_render_types
        ), "Render Manager does not recognize this render type"

        folder_name = f"{destination}{type_key}_pred"

        if folder_name not in self.folder_locs:
            self.folder_locs[folder_name] = join(self.folder_locs["base"], folder_name)
            try:
                os.mkdir(self.folder_locs[folder_name])
            except FileExistsError:
                print("File already exists - continuing")
            self.open_gif_writer(folder_name)
        elif folder_name in self.folder_locs and num == 0:
            self.images[folder_name] = []
            del self.gif_writers[folder_name]
            self.open_gif_writer(folder_name)

        if folder_name not in self.images:
            self.images[folder_name] = []

        R = self.images[type_key][num]["R"]
        T = self.images[type_key][num]["T"]

        img_manager = ImageManager(posn=num + 1, render_type=folder_name, R=R, T=T)
        img = img_as_ubyte(pred)
        img_manager._save(img, self.folder_locs[folder_name])
        self.gif_writers[folder_name].append_data(img)
        # Append to images list
        self.images[folder_name].append(img_manager._dict)

    def add_event_frame(self, count, frame):
        assert (
            "events" in self.images.keys()
        ), "Render Manager does not possess event config"

        event_manager = EventFrameManager(count, extension="png")
        event_manager._save(frame, self.folder_locs["events"])
        frame = img_as_ubyte(frame)
        # self.gif_writers["events"].append_data(frame)
        self.images["events"].append(event_manager._dict)

    def add_pred_mesh(self, mesh: Meshes, name: str):
        assert isinstance(mesh, Meshes), "Mesh should be pytorch3d mesh object"
        if "predicted_mesh" in self.__dict__.keys():
            self.predicted_mesh.append((mesh, name))
            return
        self.predicted_mesh = [(mesh, name)]

    def add_pose_plot(self, fig: plt.figure, name: str):
        if "pose_plot" in self.__dict__.keys():
            self.pose_plot.append((fig, name))
            return
        self.pose_plot = [(fig, name)]

    def set_metadata(self, meta: dict):
        assert isinstance(meta, dict), "Metadata should be a dictionary"
        if self.metadata:
            self.metadata.update(meta)
        else:
            self.metadata = meta

    def set_pred_results(self, results: dict):
        assert isinstance(results, dict), "Results should be a dictionary"
        self.pred_results = results

    def _dict(self):
        my_dict = asdict(self)
        my_dict["pred_results"] = True if self.pred_results else False
        my_dict["predicted_mesh"] = True if "predicted_mesh" in my_dict else False
        return my_dict

    def close(self):
        # close writers
        for key, gw in self.gif_writers.items():
            if isinstance(gw, str):
                continue

            gw.close()
            self.gif_writers[key] = join(
                self.folder_locs[key], f"camera_simulation_{key}.gif"
            )
        # generate json file for the render
        json_dict = self._dict()
        json_file = join(self.folder_locs["base"], "info.json")
        with open(json_file, mode="w") as f:
            json.dump(json_dict, f)

        if self.pred_results:
            results_file = join(self.folder_locs["base"], "reconstruction_results.json")
            with open(results_file, mode="w") as f:
                json.dump(self.pred_results, f)

        if "predicted_mesh" in self.__dict__.keys():
            for mesh, name in self.predicted_mesh:
                mesh_path = join(self.folder_locs["base"], f"predicted_mesh{name}.obj")
                verts, faces = mesh.get_mesh_verts_faces(0)
                save_obj(mesh_path, verts, faces)

        if "pose_plot" in self.__dict__.keys():
            for plot, name in self.pose_plot:
                plot_path = join(self.folder_locs["base"], f"pose_plot_{name}.png")
                plot.savefig(plot_path, dpi=plot.dpi)

    def rectify_paths(self, new_folder: str = "", render_folder: str = "", base_folder: str = ""):
        """Rectifies all the paths in the info.json file
        """
        self.new_folder = new_folder
        if render_folder:
            self.render_folder = render_folder
        if not base_folder:
            curr_path = __file__
            path_to_e3d = dirname(dirname(curr_path))
            dataset_name = self.folder_locs["base"].split("/")[-1]
            if not self.new_folder:
                self.new_folder = self.folder_locs["base"].split("/")[-2]
            self.base_folder = join(
                path_to_e3d, self.render_folder, self.new_folder, dataset_name
            )
        else:
            dataset_name = self.folder_locs["base"].split("/")[-1]
            self.base_folder = join(base_folder, dataset_name)

        self.folder_locs["base"] = self.base_folder

        if base_folder:
            for loc_key in self.folder_locs.keys():
                if loc_key != 'base':
                    folder = self.folder_locs[loc_key].split("/")[-1]
                    self.folder_locs[loc_key] = join(self.folder_locs['base'], folder)

            for gif_key in self.gif_writers.keys():
                folder = self.gif_writers[gif_key].split("/")[-2:]
                self.gif_writers[gif_key] = join(self.folder_locs['base'], folder[0], folder[1])


        for type_key in self.images.keys():
            for idx, img_dict in enumerate(self.images[type_key]):
                if type_key == "events":
                    manager = EventFrameManager.from_dict(img_dict)
                    old_path = manager.file_name
                    new_path = join(
                        self.folder_locs["base"], type_key, old_path.split("/")[-1]
                    )
                    manager.file_name = new_path
                else:
                    manager = ImageManager.from_dict(img_dict)
                    old_path = manager.image_path
                    new_path = join(
                        self.folder_locs["base"], type_key, old_path.split("/")[-1]
                    )
                    manager.image_path = new_path

                self.images[type_key][idx] = manager._dict

    @classmethod
    def from_path(cls, path: str):
        """
        Loads manager from absolute path to render_directory
        """
        path_to_json = join(path, "info.json")
        if not os.path.exists(path_to_json) or not os.path.isdir(path):
            raise FileNotFoundError(
                f"Incorrect path given to from_path, please ch"
                f"eck that {path} is the correct path to the render directory you're looking for"
            )
        with open(path_to_json, "r") as f:
            json_dict = json.load(f)
            ret = _from_dict(cls, json_dict)
        return ret

    @classmethod
    def from_directory(cls, dir_num: int = None, render_folder: str = "", datamode: str = ''):
        """
        Finds the most recent valid render directory or the one specified by "dir_num"
        Returns a instantiated class from that folder
        """
        if render_folder:
            cls.render_folder = render_folder
        if datamode != 'jjp':
            file_path = __file__
            path_to_e3d = dirname(dirname(file_path))
            render_folder = join(path_to_e3d, cls.render_folder)
        directory_paths = sorted(os.listdir(render_folder), reverse=True)
        for p in directory_paths:
            # Check if the info.json file is present in that directory
            full_path = join(render_folder, p)
            path_to_json = join(full_path, "info.json")
            if not os.path.exists(path_to_json) or os.path.isfile(full_path):
                continue
            num = int(p.split("-")[0])
            if dir_num == num:
                with open(path_to_json, "r") as f:
                    json_dict = json.load(f)
                    ret = _from_dict(cls, json_dict)
                return ret

        return None
