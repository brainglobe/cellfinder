# Installation

**For installation instructions in a cluster computing environment, see 
[here](misc/hpc.md)**

**Only tested on Ubuntu 16.04 and 18.04**


#### If required, install CUDA for GPU-support 
Necessary for training cell classification, highly recommended for inference.
For compatibility, we require CUDA 10.1 and cuDNN >=7.6, instructions can be
found [here](https://www.tensorflow.org/install/gpu). If you need other
versions of CUDA (for other work), please see [here](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77) for how to link CUDA versions to conda environments.

Alternatively, CUDA can be installed into your conda environment (see below).

#### Set up conda environment and install dependencies

**Recommended, allthough not necessary**
Conda environments are used to isolate python versions and packages.

**Once conda environment is set up, everything must be carried out 
in this environment**

* Download miniconda installation file 
[here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

* Run miniconda installation file, e.g.:
    
```    
bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
```
* Create and activate new minimal conda environment

``` bash
conda create --name cellfinder python=3.7
conda activate cellfinder
```    

#### Optional: Install CUDA and cuDNN into the conda environment:
*This avoids the need to install CUDA and cuDNN system-wide, but the GPU 
drivers are still required*

```bash
conda install cudatoolkit=10.1 cudnn
```

#### Install cellfinder


``` bash
pip install cellfinder

```    

**If you use a conda environment, remember to activate it
 (`conda activate cellfinder`) before using cellfinder.**

#### Download atlas and trained classification models (optional)
When cellfinder runs, the appropriate machine learning models and 
reference atlas will be downloaded (if not previously done so). If you would 
like to download in advance to save time (or if you will not have an internet
connection) please use `cellfinder_download`.

Currently, the only supported atlas is the Allen reference mouse 
brain, originally from [here](http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas).
The atlas will download into `./cellfinder/atlas`. 

In the future, other atlases will be available, and you will be able to choose
which is downloaded.

If you want to modify the cellfinder download, use:
* `--atlas-install-path` Supply a path to download the atlas elsewhere. This 
should also update the default `cellfinder.conf` file so that the correct 
atlas is sourced. Alternatively, use this command to tell cellfinder where an 
existing atlas is, to save it being downloaded twice. (Requires 20GB 
disk space)
* `--atlas download path` The path to download the atlas into. 
(Requires 1.2GB disk space). Defaults to `/tmp`.
