"""
Segmentation network evaluation step
Uses DiceCoeffLoss to evaluate performance of the network
See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
"""
#Move to top-level import to catch losses
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import DiceCoeffLoss


def eval_seg_net(net, loader):
    """
    Evaluation approach for the segmentation net
    Uses the DiceCoefficient loss defined in losses.py
    """
    
    loss = 0
    with tqdm(total=len(loader), desc='Validation', unit='batch', leave=False) as pbar:
        for X, y in loader:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            X = X.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            with torch.no_grad():
                out = net(X)
            out = torch.sigmoid(out)
            out = (out > 0.5).float()
            loss += DiceCoeffLoss().forward(out, y)
            
            pbar.update()
            
    loss = loss / len(loader)
    return loss
