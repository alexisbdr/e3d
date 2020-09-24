import random
import torch

from pytorch3d.renderer import (
    look_at_view_transform, look_at_rotation,
)
from pytorch3d.transforms import (
    Rotate, Translate, RotateAxisAngle, Transform3d
)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
TRAJECTORY AUGMENTATION
"""
def trajectory_variation(value_tensor, bounds: tuple, scale: int):
    """
        Augmentation:
            applies a continuous variation to the tensor:
                -tensor: input tensor to be augmented - modified in place
                -bounds: Tuple of bounds representing upper and lower
                -scale: int
    """
    rand_var = random.randint(bounds[0], bounds[1])
    sign = -1 if random.randint(0,1) else 1
    variation = [sign * (x / scale) for x in range(1, rand_var)]
    #Quick trick to make this variation "continuous"
    variation = variation + variation[::-1]
    idx = 0
    for d in range(len(value_tensor)):
        curr_idx = idx % len(variation)
        value_tensor[d] = value_tensor[d] + variation[curr_idx]
        idx+=1

        
def trajectory_pepper(value_tensor, bounds: tuple, scale: int):
    """
    Augmentation:
        applies irregular micro-fluctuations sampled from the bounds: 
            -tensor: input tensor to be augmented - modified in place
            -bounds: (min, max) range for random augmentation
            -scale: scaling factor 
            -sigma
    """
    
    rand_var = random.randint(bounds[0], bounds[1])
    pepper = [(x / scale) for x in range(1, rand_var)]
    idx = 0
    for d in range(len(value_tensor)):
        rand_idx = random.randint(0, 100) % len(pepper)
        rand_sign = -1 if random.randint(0, 1) else 1
        value_tensor[d] = value_tensor[d] + rand_sign * pepper[rand_idx]
        idx+=1

        
"""
TRAJECTORY GENERATORS
"""

def cam_trajectory_rotation(device: str, num_points: int = 4):
    """
    Returns: list of camera poses (R,T) from trajectory along a spherical spiral
    """
    
    shape = SphericalSpiral(
        c = 6,
        a = 3,
        t_min = 1*math.pi,
        t_max=1.05*math.pi,
        num_points=num_points)
    up = torch.tensor([[1., 0., 0.]])
    R = []
    T = []
    for cp in shape._tuples:
        cp = torch.tensor(cp).to(device)
        R_new = look_at_rotation(cp[None, :], device=device)
        T_new = -torch.bmm(R_new.transpose(1,2), cp[None, :, None])[:, :, 0]
        if not len(R) and not len(T):
            R = [R_new]
            T = [T_new]
        else:
            R.append(R_new)
            T.append(T_new)
    return (torch.stack(R)[:,0,:], torch.stack(T)[:,0,:])


def cam_trajectory(
    variation: list, 
    pepper: list,
    random_start: list,
    batch_size: int
    ):
    """
    Generates camera poses with given parameters:
        -variation ["dist","elev"] #continuous variations along this axis
        -pepper ["dist","elev"] #simulates micro-fluctuations
        -random_start ["dist","elev","azim"] #generate a random start point):
    
    """
    dist = 1.8 if not "dist" in random_start else random.randint(170,190) / 100
    elev = 35 if not "elev" in random_start else random.choice([random.randint(-55, -20), random.randint(20, 60)])
    if not "azim" in random_start:
        azim_range = [0, 300]
    else:
        random_azim_start = random.randint(0, 360)
        azim_range = [random_azim_start, random_azim_start + 360]
    
    dist = torch.tensor([dist] * batch_size)
    elev = torch.tensor([elev] * batch_size)
    #Adds continuous variation along this axis
    if "dist" in variation:
        trajectory_variation(dist, (5, 15), 100)
    if "elev" in variation:
        trajectory_variation(elev, (5, 20), 1) 
    if "dist" in pepper:
        trajectory_pepper(dist, (3, 5), 500)
    if "elev" in pepper:
        trajectory_pepper(elev, (10, 20), 100)
        
    azim = torch.linspace(azim_range[0], azim_range[1], batch_size)
    
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    
    return (R, T)
