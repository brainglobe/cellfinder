## Cellfinder installation in a cluster computing environment.

Currently cellfinder has only been written with 
[SLURM](https://slurm.schedmd.com/documentation.html) in mind. In theory, it 
should be easy enough to allow the use of any other job scheduler.

### SLURM
Based on the 
[SWC SLURM cluster](https://www.sainsburywellcome.org/web/content/scientific-computing),
 and so most of the command syntax will likely vary. Specifically, you are
  unlikely to have modules configured in exactly the same way as us.
 

#### Prepare the environment
* On our cluster, [modules](http://modules.sourceforge.net/) are only available 
on a compute node, so start an interactive job on a GPU node, and request a 
GPU for testing.

```bash
srun -p gpu --gres=gpu:1 --pty bash
```

* Load miniconda

```bash
module load miniconda
```


#### Set up conda environment and install cellfinder

* Now you can proceed as with a [local installation](../install.md)
    * Create and activate new minimal conda environment

    ``` bash
        conda create --name cellfinder python=3.7
        conda activate cellfinder
    ```    

    * Install CUDA and cuDNN

    ``` bash
        conda install cudatoolkit=10.1 cudnn
    ```
  
    * Install cellfinder

    ``` bash
        pip install git+https://github.com/adamltyson/cellfinder
    ```
    
* Check that tensorflow and CUDA are configured properly:
    ```bash
    python
    ```
    ```python
       import tensorflow as tf
       tf.test.is_gpu_available()
    ```
    If you see something like this, then all is well.
    
    ```bash
        2019-06-26 10:51:34.697900: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX512F
        2019-06-26 10:51:34.881183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
        name: TITAN RTX major: 7 minor: 5 memoryClockRate(GHz): 1.77
        pciBusID: 0000:2d:00.0
        totalMemory: 23.62GiB freeMemory: 504.25MiB
        2019-06-26 10:51:34.881217: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
        2019-06-26 10:51:35.251465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
        2019-06-26 10:51:35.251505: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
        2019-06-26 10:51:35.251511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
        2019-06-26 10:51:35.251729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/device:GPU:0 with 195 MB memory) -> physical GPU (device: 0, name: TITAN RTX, pci bus id: 0000:2d:00.0, compute capability: 7.5)
        True
     ```
* **End your interactive job**
```bash
exit
```

#### Run cellfinder
Allthough you can run cellfinder interactively, it is better to submit a batch 
job.

* Write the job submission script. An example can be found 
[here](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/cellfinder_sbatch.sh). 
If possible, set the output directory to local, fast scratch storage.


* Submit the job to the job scheduler
```bash
sbatch cellfinder_sbatch.sh
```

* If you use the example script, you will recieve an email when the job is 
done. To watch the progress, log onto a node with the same storage drive 
mounted and run:
```bash
watch tail -n 100 /path/to/cellfinder_log.log
```

* Copy the results from the storage platform.