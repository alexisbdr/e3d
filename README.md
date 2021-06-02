# E3D: Event-Based Shape Reconstruction

## Running E3D

### Installing Pytorch3D
* [Linux - Ubuntu16+ or/ CentOS 7]
* [Python 3.6+]
* [Pytorch 1.0+]
* [gcc & g++ 4.9+]
* [fvcore]
* [CUDA 9.2+ (If CUDA is to be used)]

Create an [Anaconda](https://docs.anaconda.com/anaconda/install/) Environment:
```
conda env create -n pytorch3d python=3.7 --file env.yml
conda activate pytorch3d
```

Install a version of pytorch and torchvision suitable for your environment, see [Pytorch](https://pytorch.org/) for instructions. For example:
```
#CPU Only
conda install pytorch torchvision cpuonly -c pytorch
#CUDA 10.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Install Pytorch3d with CUDA Support (Change for your cuda version):
```
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install -c pytorch3d pytorch3d
```
If you run into an "Unsatisfiable Error" with the current version of your CUDA driver then you should install Pytorch3D nighty build:
```
conda install -c pytorch3d-nightly pytorch3d
```

Installing Pytorch3D without CUDA Support:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Other Dependencies
Installing [RPG Vid2e](https://github.com/alexisbdr/rpg_vid2e) for the event generator. Cmake required to build the code
```
git clone https://github.com/alexisbdr/rpg_vid2e.git --recursive
conda install -y -c conda-forge opencv tqdm scikit-video eigen boost boost-cpp pybind11
pip install -e . 
```

Installing [PMO](https://github.com/alexisbdr/photometric-mesh-optim.git)
```
git clone https://github.com/alexisbdr/photometric-mesh-optim.git
```

## Pre-Trained Models

Category | Drive Link 
------------- | ------------- 
car  | [link](https://drive.google.com/file/d/1UOdLMux0nr4S7QzST1hjyJlgeASu8JR9/view?usp=sharing)
chair | [link](https://drive.google.com/file/d/1uQXTkqTj38UYaMY5Zk8IAZJvFTebjWsf/view?usp=sharing)
dolphin (baseline) | [link](https://drive.google.com/file/d/1zGdw7QoPtytwQDfbYirZaWC7Ctw69kFI/view?usp=sharing) 
dolphin (fine-tuned) | [link](https://drive.google.com/file/d/1VrA8_Dgdto-JxexaT6BwCTFw7NaR3oWT/view?usp=sharing)
------------- | ------------- 
car (PMO - Events) | [link](https://drive.google.com/file/d/1klYc0SkwBBGLUTJd64JjO3gJbxiPg1cp/view?usp=sharing)
chair (PMO - Events) | [link](https://drive.google.com/file/d/1o3Dst-QRZR15Ph6YuwVkotXZVXUu1_0r/view?usp=sharing)

## Datasets
Toy datasets of both car and chair are provided with the code to ease reproducibility. We recommend running with the toy datasets (as described below) to reproduce the results seen in the paper/

You will need at least 20G of space to download the full datasets

The datasets must be downloaded to data/renders

Name  | Category | Drive Link 
------------- | ------------- | -----------------
test_car_subset | car | [link](https://drive.google.com/file/d/1wf885mLpn5Ixk9t1xc3_uKW3qBgq5Jt9/view?usp=sharing)
test_chair_subset | chair | [link](https://drive.google.com/file/d/1KqJTxctb_tWnukxBUduo69XSOP5QuGOC/view?usp=sharing)
train_car_shapenet | car | [link](https://drive.google.com/file/d/1fMzvSkENq0lfqC5c6C3g34swufo3NtmV/view?usp=sharing)
test_car_shapenet | car | [link](https://drive.google.com/file/d/1fz0Hb9WYaOB5K7DOJw3Icoc2KR6Ys2SU/view?usp=sharing)
train_chair_shapenet | chair | [link](https://drive.google.com/file/d/1mpyYI99KmkRG72oFYB5xr_vPDOtle74i/view?usp=sharing)
test_chair_shapenet | chair | [link](https://drive.google.com/file/d/1rXqfLmqn8yo_txmL5Mft5FmwqxERnTEx/view?usp=sharing)
train_dolphin | dolphin | [link](https://drive.google.com/file/d/1PjR3j4CmmQhN84NohQXmT48MjWvFQSKW/view?usp=sharing)
test_dolphin | dolphin | [link](https://drive.google.com/file/d/1TzTdCihnUlnx1-cDL1CrW_0mAJXUXjTX/view?usp=sharing)

### Running pre-trained models
Default mesh reconstruction parameters are in mesh_reconstruction/params.py
Car:
```
python predict.py --gpu 0 --model model_checkpoints/car_shapenet.pth --path data/renders/test_car_subset
```
Chair:
```
python predict.py --gpu 0 --model model_checkpoints/chair_shapenet.pth --path data/renders/test_chair_subset
```

### Training
You can find the default training parameters in segpose/params.py
Car:
```
python train-segpose.py --gpu 0 -t data/renders/train_car_shapenet --name car_shapenet
```
Chair:
```
python train-segpose.py --gpu 0 -t data/renders/train_chair_shapenet --name chair_shapenet
```

### Generating a synthetic event dataset
Default parameters are in synth_dataset/params.py
```
cd synth_dataset
python generate_dataset.py --gpu 0 --name test_car --category car
```
The dataset will be generated in data/renders by default


### Contributing
Any contributions are much appreciated!!
The repository uses pre-commit to clean the code before committing. To install run:
```
conda install -c conda-forge pre-commit
pre-commit install
```
This should apply the pre-commit hooks. If this is your first time contributing to an open source project, follow the guidelines [here](https://github.com/firstcontributions/first-contributions) 
