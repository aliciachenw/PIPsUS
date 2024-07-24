# Install pytorch 1.13.1 cuda 11.6
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

MAINTAINER Wanwen Chen

ARG https_proxy
ARG http_proxy

ENV TZ=America/Los_Angeles \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl \
    vim

RUN apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip

RUN pip3 install --upgrade pip

# Install dependency
RUN pip3 --no-cache-dir install \
    numpy \
    SimpleITK \
    scipy \
    pillow

RUN pip3 install einops==0.6.1 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.3 \
    opencv-python==4.8.0.76 \
    scikit-image==0.21.0 \
    albumentations==1.3.1 \
    tensorboardX==2.6.2.2 \
    fire==0.5.0 \
    moviepy==1.0.3 \
    prettytable==3.9.0

RUN pip3 install tensorboard

# Fix opencv lib error
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Set the library path to use cuda and cupti
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN apt-get update