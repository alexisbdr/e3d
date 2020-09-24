import argparse
import os
from os.path import join, abspath
import sys
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torchvision import transforms

from segmentation import UNet
from segmentation.params import Params
from segmentation.dataset import EvMaskDataset
from utils.manager import RenderManager
from utils.visualization import plot_img_and_mask

def get_args():
    
    parser = argparse.ArgumentParser(description='EvUnet Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', dest='model_cpt', type=str, default=Params.model_cpt,
                        help='Load model from a .pth file')
    parser.add_argument('-p', '--path', dest='pred_dir', type=str, default=Params.pred_dir, 
                        help='Path to prediction directory')
    parser.add_argument('-t', '--threshold', dest='threshold_conf', default=Params.threshold_conf,
                        help='Probability threshold for masks')
    
    return parser.parse_args()


def predict(unet, img, threshold, img_size):
    """Runs prediction for a single PIL Image
    """
    ev_frame = torch.from_numpy(EvMaskDataset.preprocess(img, img_size))
    ev_frame = ev_frame.unsqueeze(0).to(device=device, dtype=torch.float)
            
    with torch.no_grad():

        mask_pred = unet(ev_frame)
        probs = torch.sigmoid(mask_pred).squeeze(0).cpu()
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy() > threshold

    plot_img_and_mask(img, full_mask)
    
    return (full_mask * 255).astype(np.uint8)
    
                

def main(unet, params):
    
    
    if not params.pred_dir and not os.path.exists(params.pred_dir):
        raise FileNotFoundError("Prediction directory has not been set or the file does not exist, please set using cli args or params")
    
    pred_folders = [join(params.pred_dir, f) for f in os.listdir(params.pred_dir)]
    for p in pred_folders:
        try:
            manager = RenderManager.from_path(p)
        except FileNotFoundError:
            continue
        for idx in range(len(manager)):
            
            ev_frame = manager.get_event_frame(idx)
            logging.info(f"Starting prediction for frame #{idx}")
            mask_pred = predict(unet, ev_frame, params.threshold_conf, params.img_size)
            
            manager.add_pred(idx, mask_pred, "silhouette")

        manager.close()
    
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    #Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0": torch.cuda.set_device()

    logging.info(f"Using {device} as computation device")
    
    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)
    params.device = device
        
    try:
        model = UNet.load(params)
        logging.info("Loaded UNet from params")
        main(model, params)
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating prediction run")
        sys.exit(0)
