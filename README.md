# E3D: Recovering Shape from Events

## Running E3D on AWS EC2
### Setting up the environment
#### OS/Hardware Requirements
* [Ubuntu 16+ (Tested on Ubuntu 18.04)]
* [Nvidia GPU w/ 4Gb+ Memory]
* [~50GB of SSD storage]
* [Tensorflow-gpu 2.0+]
* [OpenCV3]

Follow these steps:

#### [SUGGESTED] EC2 Instance & Deep Learning AMI
* [Deep Learning Base AMI](https://aws.amazon.com/marketplace/pp/B07Y3VDBNS)
* Instance Type:
    *testing: g4dn.xlarge
    * training: g4dn.4xlarge
* Storage: 
    * 50Gb of ssd

### OPTION 1 - [DOCKER]()
```
docker pull abaudron0215/warehouse-anomaly
docker run --gpus all -it abaudron0215/warehouse-anomaly bash
```

### OPTION 2 - MANUAL INSTALL WITH ANACONDA
#### 1. Install Anaconda for Ubuntu
```
## You can visit (https://www.anaconda.com/distribution/) to install a different version of Anaconda
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

## Check the sum 
sha256sum Anaconda3-2020.02-Linux-x86_64.sh

## Run the script and answer 'yes' to everything
bash Anaconda3-2020.02-Linux-x86_64.sh
```

#### 2. Create and activate the Tensorflow environment
```
source ~/.bashrc
conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu
conda install pillow matplotlib
```
    
#### 3. Clone the repo
```
git clone https://github.com/alexisbdr/warehouse-anomaly
```
