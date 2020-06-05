import os
import sys
from os.path import join, abspath, dirname
#Move path to top level directory
sys.path.insert(1, (abspath(join(dirname(__file__), "../"))))

from dataclasses import dataclass, field, asdict
import numpy as np
import time
import json
from PIL import Image
import imageio
from skimage import img_as_ubyte
import torch

import logging

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
    R: torch.tensor = field(default_factory=torch.tensor)
    T: torch.tensor = field(default_factory=torch.tensor)

    extension: str = "jpg"

    @property
    def _dict(self):
        self.R = json.dumps(self.R.tolist())
        self.T = json.dumps(self.T.tolist())
        return asdict(self)

    @property
    def _load(self):
        return imageio.imread(self.image_path)
        #return Image.open(path)

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
        mod = dict_in.copy()
        use_dict = {}
        for entry in mod:
            if entry in cls.__annotations__:
                if entry == "R" or entry == "T":
                    dict_in[entry] = torch.tensor(json.loads(dict_in[entry]))
                use_dict[entry] = dict_in[entry]
        return cls(**use_dict)

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

    #List of render types
    types: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    #Internally managed
    count: int = 0 #This is a count of poses not total images
    folder_locs: dict = field(default_factory=dict)
    formatted_utc_ts: str = ""
    gif_writers: dict = field(default_factory=dict)

    def __post_init__(self):
        #Timestamp format
        curr_struct_UTC_ts = time.gmtime(time.time())
        self.formatted_utc_ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        self.folder_locs['base'] = f"data/renders/{self.mesh_name}_{self.formatted_utc_ts}"
        logging.info(f"Render Manager started in base file {self.folder_locs['base']}")
        for t in self.types:
            print(t)
            if t not in self.allowed_render_types:
                raise TypeError(f"RenderManager: Wrong image type set in init, an image type must be one of: {ImageManager.allowed_types}")
            #Create a folder for each type
            self.folder_locs[t] = join(self.folder_locs['base'], t)
            os.makedirs(self.folder_locs[t], exist_ok=True)
            #Create a gif writer for each type
            gif_t_loc = join(self.folder_locs[t], f"camera_simulation_{t}.gif")
            #TODO[ALEXIS] Need to play around with this duration parameter
            gif_t_writer = imageio.get_writer(gif_t_loc, mode="I", duration=1)
            self.gif_writers[t] = gif_t_writer
            #Create an image storage for the image type
            self.images[t] = []

    @property
    def allowed_render_types(self):
        return ["silhouette", "shaded", "textured"]

    def add_images(self, count, imgs_data, R, T):
        #Create ImageData class for each type of image
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

    def set_metadata(self, meta):
        self.metadata = meta

    def close(self):
        #close writers
        for key, gw in self.gif_writers.items():
            gw.close()
            self.gif_writers[key] = join(self.folder_locs[key], f"camera_simulation_{key}.gif")
        #generate json file for the render
        json_dict = asdict(self)
        json_file = join(self.folder_locs['base'], "info.json")
        with open(json_file, mode="w") as f:
            json.dump(json_dict, f)


if __name__ == "__main__":
    rm = RenderManager(
        mesh_name = "teapot",
        types = ["shaded"]
    )


