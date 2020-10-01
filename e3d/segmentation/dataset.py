import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.manager import RenderManager

class EvMaskDataset(Dataset):
    
    def __init__(self, 
                 dir_num: int,
                 params,
                 transforms: list = []):
        
        self.img_size = params.img_size
        self.transforms = transforms
        self.render_manager = RenderManager.from_directory(
            dir_num = dir_num,
            render_folder = params.train_dir
        )
 
    def __len__(self):
        return len(self.render_manager)
    
    @classmethod
    def preprocess(self, img: Image, img_size):
        """Resize and normalize the images to range 0, 1
        """
        img = img.resize(img_size)
        
        img_np = np.array(img)
        
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)

        # HWC to CHW
        img_trans = img_np.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        return img_trans
    
    def __getitem__(self, index: int):
        
        mask = self.render_manager.get_image("silhouette", index)
        
        event_frame = self.render_manager.get_event_frame(index)
        
        assert mask.size == event_frame.size, \
            "Mask and event frame must be same size"
        
        mask = torch.from_numpy(self.preprocess(mask, self.img_size)).type(torch.FloatTensor)
        event_frame = torch.from_numpy(self.preprocess(event_frame, self.img_size)).type(torch.FloatTensor)
        
        return event_frame, mask
           