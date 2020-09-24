import configparser
from dataclasses import dataclass, asdict

from torch import optim

from utils.pyutils import _from_dict

@dataclass
class Params:
    
    #Folder Params
    train_dir: str = "data/renders/train_dolphin"
    pred_dir: str = "data/renders/test_dolphin"
    model_cpt: str = ""
    config_file: str = ""
    
    #Image Params
    img_size: tuple = (560, 560)
    
    #Training Params
    optimizer = optim.RMSprop #optim.Adam
    batch_size: int = 5
    epochs: int = 15
    learning_rate: float = .0001
    val_split: float = .15
        
    #Unet Params
    n_channels: int = 1
    n_classes: int = 1
    bilinear: bool = False
        
    threshold_conf: float = 0.5
    
    def __post_init__(self):
        if self.config_file:
            with open(self.config_file) as f:
                dict_in = json.load(f)
                for key, value in dict_in.items():
                    if key in self._annotations__:
                        setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, dict_in: dict):
        return _from_dict(cls, dict_in)
            