# Troubleshooting

As cellfinder relies on a number of third party libraries (notably 
[TensorFlow](https://www.tensorflow.org/),
 [CUDA](https://developer.nvidia.com/cuda-zone) and 
 [cuDNN](https://developer.nvidia.com/cudnn)) there may be issues while 
 running the software.

## Error messages

#### OSError: [Errno 24] Too many open files
```bash
OSError: [Errno 24] Too many open files
```
This is likely because your default limits are set too low (allthough this
should probably be prevented from happening at all see 
[here](https://github.com/adamltyson/cellfinder/issues/8)). To fix this, 
follow the instructions 
[here](https://easyengine.io/tutorials/linux/increase-open-files-limit/). If 
for any reason you don't want to or can't change the system-wide limits, 
running `ulimit -n 60000` before running cellfinder should work. This setting 
will persist for the present shell session, but will have to repeated if you 
open a new terminal.

#### libcublas.so.9.0: cannot open shared object file: No such file or directory
```bash
ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory
Failed to load the native TensorFlow runtime.
See https://www.tensorflow.org/install/errors
for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
```

If you see an error like this, it is likely that you don't have CUDA installed, 
or it isn't added to your path (so tensorflow can't find it). If you have a 
CUDA 10 version of tensorflow-gpu installed, you may instead see:
```bash
ImportError: libcublas.so.10.0: cannot open shared object file: No such file or directory
```

If you're not sure which version of CUDA you need, please see 
[here](misc/CUDA10.md). If you don't have one of these newer GPUs, it is easier
to use CUDA 9.

If CUDA and cuDNN is definately installed (following instructions 
[here](https://www.tensorflow.org/install/gpu)). Then please make sure that 
they are added to the path. If for example your version of CUDA is located at 
`/usr/local/cuda-10.0`, then run:

```bash
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```

To save doing this each time you run cellfinder, you can add these lines to 
e.g. your [`~/.bashrc`](https://www.maketecheasier.com/what-is-bashrc/) file. 
If you have multiple versions of CUDA, or don't want to edit your `.bashrc` 
file, see [here](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77) 
for how to only add to the path for specific conda environments.



#### INFO:tensorflow:Error reported to Coordinator: Failed to get convolution algorithm
```bash
INFO:tensorflow:global_step/sec: 0
2019-05-17 13:04:53.550306: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2019-05-17 13:04:53.565467: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
INFO:tensorflow:Error reported to Coordinator: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
```

If you see an error like this, it's likely to be one of two things, either 
your GPU memory is full, or an issue with your CUDA and cuDNN version.

Your GPU memory may be full if it is still being used by 
another process. To test this, run `nvidia-smi`, and you will see something 
like this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.56       Driver Version: 418.56       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  TITAN RTX           On   | 00000000:2D:00.0  On |                  N/A |
| 41%   38C    P2    55W / 280W |  23408MiB / 24187MiB |      4%      Default |
+-------------------------------+----------------------+----------------------+
```
The bit to look for is the memory use (`23408MiB / 24187MiB`), is this is 
nearly full (like the example) then find the culprit:
```
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     37793      C   ...miniconda3/envs/cellfinder/bin/python 23408MiB   |
+-----------------------------------------------------------------------------+
```

In this case, a previous run of cellfinder hasn't completed. Either wait for 
it to run, or cancel it with CTRL+C (in the cellfinder terminal).


Alternatively, your version of CUDA and cuDNN may be not compatible with 
tensorflow 2.0. You can update them by following the instructions 
[here](https://www.tensorflow.org/install/gpu)or by installing them into your 
conda environment:
```bash
conda install cudatoolkit cudnn
```

#### SyntaxError: invalid syntax (logging)
```bash
  File "/home/adam/projects/cellfinder/cellfinder/tools/tools.py", line 444
    logging.debug(f"Free memory is: {free_mem} bytes.")
                                                     ^
SyntaxError: invalid syntax
```

If you see an error like this, with the second line starting with something
like `logging.debug(f"`, `logging.info(f"` or `print(f"`, then you likely 
have an unsupported (<3.6) version of python. Use conda or pip to install
python 3.6.



## Things that look like errors, but aren't:
Most things that are actually errors will interrupt cellfinder, and the program
won't run. Other things will get logged with an `ERROR` or a `WARNING` flag 
and will get printed to the console in addition to the log file.

A number of third party modules may raise their own errors. As long as you 
understand what they mean, they can usually be safely ignored.

#### Can't find openCV
```bash
CRITICAL:tensorflow:Optional Python module cv2 not found, please install cv2 and retry if the application fails.
```
Tensorflow thinks this is critical, it's not.

#### CPU instruction sets
```bash
tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
```
Unless you built tensorflow from source, something like this will come up. 
It'll still work fine