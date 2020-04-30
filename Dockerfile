
# System
FROM ubuntu:18.04
LABEL maintainer "Eric Chen - ericrc@mit.edu"
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python-numpy python-dev python-pip python-opengl \
    python3-dev python3-pip \
    make \
    cmake \
    curl \
    gcc \
    git \
    golang \
    fceux \
    ffmpeg \
    tmux \
    htop \
    iputils-ping \
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
    mongodb-server \
    xvfb \
    xorg-dev

# Python Dependencies
RUN pip install numpy vizdoom go_vncdriver openpyxl omgifol
COPY ./curiosity/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Install pycolab locally
COPY ./pycolab /pycolab
WORKDIR /pycolab
RUN pip install -e /pycolab

# Install mazeworld locally
COPY ./mazeworld /mazeworld
WORKDIR /mazeworld
RUN pip install -e /mazeworld

# Install doom locally
COPY ./vizdoomgym /vizdoomgym
WORKDIR /vizdoomgym
RUN pip install -e /vizdoomgym

# Install mario locally
COPY ./gym-super-mario /gym-super-mario
WORKDIR /gym-super-mario
RUN pip install -e /gym-super-mario

WORKDIR /






