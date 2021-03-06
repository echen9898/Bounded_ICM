
# System Dependencies
FROM ubuntu:18.04
LABEL maintainer "Eric Chen - ericrc@mit.edu"
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y \
    python-numpy python-dev python-pip python-opengl \
    python3-dev python3-pip \
    make \
    cmake \
    gcc \
    git \
    golang \
    fceux \
    ffmpeg \
    tmux \
    htop \
    lsof \
    nano \
    zlib1g-dev \
    libjpeg-dev \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    libjpeg-turbo8-dev \
    wget \
    unzip \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libxxf86vm-dev \
    libgl1-mesa-dev \
    mesa-common-dev \
    xvfb \
    xorg-dev

# Python Dependencies
RUN pip install numpy vizdoom go_vncdriver
COPY ./curiosity/requirements.txt /requirements.txt
RUN pip install -r ./requirements.txt
COPY ./vizdoomgym /vizdoomgym
WORKDIR /vizdoomgym
RUN pip install -e /vizdoomgym
WORKDIR /