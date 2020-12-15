from dataclasses import dataclass, field

from utils.params_base import ParamsBase

shapenet_synset_map = {"car": "02958343", "chair": "02691156", "plane": "03001627"}


@dataclass
class Params(ParamsBase):

    gpu_num: int

    name: str
    config_file: str = ""

    # Dataset Generation Params
    img_size: tuple = (280, 280)
    sigma_hand: float = 0.15
    mesh_translation: float = 0.1
    variation: list = field(default_factory=lambda: ["dist", "elev"])
    pepper: list = field(default_factory=lambda: ["elev"])
    random_start: list = field(default_factory=lambda: ["dist", "azim"])

    ## Dataset Size
    mini_batch: int = 72
    batch_size: int = 360
    data_batch_size: int = 45
    mesh_iter: int = 4  # Iterations of data loop per mesh

    show_frame: bool = False

    # Mesh info
    mesh_path: str = "../data/meshes/dolphin/dolphin.obj"
    ## Shapenet info
    shapenet: bool = True
    category: str = "car"
    synsets: dict = field(default_factory=lambda: shapenet_synset_map)
    shapenet_path: str = "../data/ShapeNetCorev2"
