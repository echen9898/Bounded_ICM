
# System Dependencies
FROM ubuntu:18.04
RUN apt-get -y update
RUN apt-get install -y python-numpy python-dev python-pip cmake zlib1g-dev libjpeg-dev xvfb \
xorg-dev python-opengl libboost-all-dev libsdl2-dev swig python3-dev \
python3-venv python3-pip make golang libjpeg-turbo8-dev gcc wget unzip git fceux ffmpeg \
tmux libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev \
libxxf86vm-dev libgl1-mesa-dev mesa-common-dev nano tmux htop

# Python Dependencies
RUN pip install numpy vizdoom go_vncdriver
COPY ./curiosity/requirements.txt /requirements.txt
RUN pip install -r ./requirements.txt
COPY ./vizdoomgym /vizdoomgym
WORKDIR /vizdoomgym
RUN pip install -e /vizdoomgym
WORKDIR /