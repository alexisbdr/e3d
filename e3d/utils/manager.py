import os
from os.path import join, abspath, dirname
import sys
sys.path.insert(0, abspath(join("..", dirname(__file__))))
print(sys.path)

from copy import deepcopy
from typing import List
from dataclasses import dataclass, field, asdict
import numpy as np
import time
import json
from PIL import Image
import imageio
from skimage import img_as_ubyte
import torch

import logging

from utils.pyutils import from_dict

def default_tensor():
    return torch.tensor([])

@dataclass
class EventFrameManager:
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
            return Image.open(self.file_name)

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
        return from_dict(cls, dict_in)

@dataclass
class ImageManager:
    """
    Contains information about a single rendered image
    Serializes/De-Serializes the image using imageio test
    """
    #[required]: position in the render
    posn: int = 0

    image_path: str = ""

    #one of "silhouette", "shaded" or "textured"
    render_type: str = ""

    #Camera pose at that render point
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
        data = Image.open(self.image_path)
        if self.render_type == "phong":
            data = self.gray(data)
        return data

    def gray(self, img):
        if isinstance(img, Image):
            return img.convert('L')
        return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    def _save(self, image_data, f_loc, img_format="png"):
        if img_format=="jpg":
            self.extension = img_format
            img_format = "JPEG-PIL"
            extra_args = {}
        elif img_format=="png":
            self.extension = img_format
            img_format = "PNG"
            #img_format = img_format.upper()
            #Lossy conversion from float32 to uint8
            """
            info = np.finfo(image_data.dtype)
            #Normalize the image
            image_data /= info.max
            image_data *= 255
            image_data = image_data.astype(np.uint8)
            """
            #Lowered the compression level for improved performance
            #Refer to this issue https://github.com/imageio/imageio/issues/387
            extra_args = {"compress_level" : 3}
        elif img_format=="tif":
            self.extension = img_format
            img_format = img_format.upper()
            extra_args = {}
        img_file = f"{self.posn}_{self.render_type}.{self.extension}"
        self.image_path = join(f_loc, img_file)
        imageio.imwrite(self.image_path, image_data, format=img_format, **extra_args)

    @classmethod
    def from_dict(cls, dict_in):
        return from_dict(cls, dict_in)

@dataclass
class RenderManager:
    """
    Manages incoming rendered images and places them into catalog folder (numbered 0 - n)
    Creates a json file with meta information about the folder and details about the images
    Creates a gif of the render
    """
    mesh_name: str = ""
    #List of paths on disk - storing the dataclass here might make it too large (to test)
    images: dict = field(default_factory=dict)

    #Trajectory
    R: list = field(default_factory=list)
    T: list = field(default_factory=list)

    #List of render types
    types: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    #Internally managed
    count: int = 0 #This is a count of poses not total images
    folder_locs: dict = field(default_factory=dict)
    formatted_utc_ts: str = ""
    gif_writers: dict = field(default_factory=dict)

    render_folder: str = "data/renders/"
    new_folder: str = ""
        
        
    def init(self):
        """Initialization step for the manager
            1. creates folder architecture on disk
            2. initializes gif writers and image dict
        """
        #Timestamp format
        curr_struct_UTC_ts = time.gmtime(time.time())
        self.formatted_utc_ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        #Construct absolute
        curr_path = __file__
        path_to_e3d = dirname(dirname(curr_path))
        self.base_folder = join(path_to_e3d, self.render_folder, self.new_folder)
        os.makedirs(self.base_folder, exist_ok = True)
  
        nums = [0]
        for f in os.listdir(self.base_folder):
            try:
                nums.append(int(f.split("-")[0]))
            except:
                continue
        render_posn = max(nums) + 1
        self.folder_locs['base'] = os.path.join(self.base_folder, f"{render_posn:03}-{self.mesh_name}_{self.formatted_utc_ts}")
        logging.info(f"Render Manager started in base file {self.folder_locs['base']}")
        for t in self.types:
            if t not in self.allowed_render_types:
                raise TypeError(f"RenderManager: Wrong image type set in init, an image type must be one of: {self.allowed_render_types}")
            #Create a folder for each type
            self.folder_locs[t] = join(self.folder_locs['base'], t)
            os.makedirs(self.folder_locs[t], exist_ok=True)
            self.open_gif_writer(t)
            self.images[t] = []

    @property
    def _trajectory(self) -> tuple:
        R = torch.stack(([torch.tensor(r) for r in self.R]))[:,0,:]
        T = torch.stack(([torch.tensor(t) for t in self.T]))[:,0,:]
        return (R, T)

    @property
    def allowed_render_types(self):
        return ["silhouette", "phong", "textured", "events"]

    def open_gif_writer(self, t: str, duration: float = .2):
        if t in self.gif_writers:
            return
        gif_t_loc = join(self.folder_locs[t], f"camera_simulation_{t}.gif")
        gif_t_writer = imageio.get_writer(gif_t_loc, mode="I", duration=duration)
        self.gif_writers[t] = gif_t_writer

    def __len__(self):
        return self.count

    def get_image(self, type_key: str, index: int) -> Image:
        """Returns a torch tensor of the loaded image at the given index
        """
        assert index >= 0 and index <= self.count, \
            f'Index {index} out of bounds for image type {type_key}'
        assert type_key in self.images, \
            f'Incorrect Type {type_key}'
        img_dict = deepcopy(self.images[type_key][index])
        img_manager = ImageManager.from_dict(img_dict)
        return img_manager._load

    def _images(self, type_key:str = "phong") -> list:
        """Returns a stacked tensor of image tensors
        """
        images_data = [
            torch.from_numpy(
                np.array(self.get_image(type_key, num)))
            for num in self.count]
        return torch.stack(images_data)

    def get_event_frame(self, index) -> Image:
        """Returns a torch tensor of the loaded event frame at the given index
        """
        assert index >= 0 and index <= self.count, \
            f'Index {index} out of bounds for events'
        event_dict = deepcopy(self.images["events"][index])
        event_manager = EventManager.from_dict(event_dict)
        return event_manager._load

    def _events(self) -> list:
        """Returns torch tensor of event frames for the given key
        """
        assert "events" in self.images, \
            "No events available for this Render Manager"
        event_data = [
            torch.from_numpy(
                np.array(self.get_event_frame(num))) 
            for num in len(self.images["events"])]
        return torch.stack(event_data)

    def add_images(self, count, imgs_data, R, T):
        #Create ImageData class for each type of image
        R = R.tolist()
        T = T.tolist()
        for img_type in imgs_data.keys():
            if img_type not in self.images.keys():
                raise TypeError(f"RenderManager: wrong render type {img_type}")
            img_manager = ImageManager(
                posn=count,
                render_type = img_type,
                R = R,
                T = T
            )
            img_manager._save(imgs_data[img_type], self.folder_locs[img_type])
            #Append to gif writer
            img = img_as_ubyte(imgs_data[img_type])
            self.gif_writers[img_type].append_data(img)
            #Append to images list
            self.images[img_type].append(img_manager._dict)
        self.count += 1
        if not len(R) and not len(T):
                self.R = R
                self.T = T
        else:
            self.R.append(R)
            self.T.append(T)

    def add_event_frame(self, count, frame):
        assert "events" in self.images.keys(),\
            "Render Manager does not possess event config"
        event_manager = EventFrameManager(count, extension="png")
        event_manager._save(frame, self.folder_locs["events"])
        frame = img_as_ubyte(frame)
        self.gif_writers["events"].append_data(frame)
        self.images["events"].append(event_manager._dict)

    def set_metadata(self, meta):
        self.metadata = meta

    def _dict(self):
        return asdict(self)

    def close(self):
        #close writers
        for key, gw in self.gif_writers.items():
            gw.close()
            self.gif_writers[key] = join(self.folder_locs[key], f"camera_simulation_{key}.gif")
        #generate json file for the render
        json_dict = self._dict()
        json_file = join(self.folder_locs['base'], "info.json")
        with open(json_file, mode="w") as f:
            json.dump(json_dict, f)

    @classmethod
    def from_directory(cls, dir_num: int = None):
        """
        Finds the most recent valid render directory or the one specified by "dir_num"
        Returns a instantiated class from that folder
        """
        file_path = __file__
        path_to_e3d = dirname(dirname(file_path))
        render_folder = join(path_to_e3d, cls.render_folder)
        directory_paths = sorted(os.listdir(render_folder), reverse=True)
        for p in directory_paths:
            #Check if the info.json file is present in that directory
            full_path = join(render_folder, p)
            path_to_json = join(full_path, "info.json")
            if not os.path.exists(path_to_json):
                continue
            num = int(p.split("-")[0])
            if dir_num is None or dir_num == num:
                break
        with open(path_to_json, 'r') as f:
            json_dict = json.load(f)
            ret = from_dict(cls, json_dict)
            #ret.images = json_dict['images']
        return ret






