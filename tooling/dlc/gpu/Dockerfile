FROM nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04

#Dockerfile originally borrowed and modified from https://github.com/saiprashanths/dl-docker

MAINTAINER Ed Henry <edward.m.henry@gmail.com>

ARG KERAS_VERSION=1.1.0

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libopenjpeg5 \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		python-pip \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

# Install python3
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel


# Install PyTorch
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
RUN pip3 install torchvision

# Add SNI support to Python
RUN pip install \
		--upgrade pandas \
		--upgrade six \
		seaborn \
		futures \
		appdirs \
		packaging \
		pyopenssl \
		ndg-httpsclient \
		pyasn1 \
		functional

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-sklearn \
		python-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip install --upgrade ipython && \
	pip install \
		Cython \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python -m ipykernel.kernelspec


# Install TensorFlow
RUN pip install tensorflow-gpu

# Install Keras
RUN pip install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

# Install OpenAI Gym
WORKDIR /opt
RUN git clone https://github.com/openai/gym.git && pip install -e './gym[all]'

# Expose Ports for TensorBoard (6006), Ipython (8888), VNC (5900)
EXPOSE 6006 8888 5900

WORKDIR "/root"
CMD ["/bin/bash"]
