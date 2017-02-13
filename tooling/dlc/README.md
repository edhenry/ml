# [Deep Learning Container](https://hub.docker.com/r/edhenry/dlc/)

This image has the following dependencies installed :

### v 0.1-cpu

* [TensorFlow v 0.11 (CPU Compiled)](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md#release-0110)
* [Hadoop 2.7.3](https://hadoop.apache.org/docs/r2.7.3/)
* [Keras 1.1.0](https://github.com/fchollet/keras/releases/tag/1.1.0)
* [Jupyter](http://jupyter.org/)
* Numpy, Scipy, Pandas, Scikit-learn, Matplotlib


### v 0.1-gpu

**Will need to be run using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for nvidia GPUs**

* [TensorFlow v 0.11 (GPU Compiled) ](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md#release-0110)
* [Hadoop 2.7.3](https://hadoop.apache.org/docs/r2.7.3/)
* [Keras 1.1.0](https://github.com/fchollet/keras/releases/tag/1.1.0)
* [Jupyter](http://jupyter.org/)
* [CUDA 8](https://developer.nvidia.com/cuda-toolkit)
* [cuDNN](https://developer.nvidia.com/cudnn)
* Numpy, Scipy, Pandas, Scikit-learn, Matplotlib

### Create a container from this image
#### Start Jupyter - CPU Image

To start the jupyter notebook server when instantiating a container from this image, issue the following command :

```docker run edhenry/dlc:<tag> /bin/sh -c "ipython notebook --ip 0.0.0.0"```

#### Start Jupyter - GPU Image (nvidia environment)

```nvidia-docker run edhenry/dlc:<tag> /bin/sh -c "ipython notebook --ip 0.0.0.0"```

#### Quick How-To for utilizing the native TensorFlow HDFS support 

https://www.tensorflow.org/how_tos/hadoop/
