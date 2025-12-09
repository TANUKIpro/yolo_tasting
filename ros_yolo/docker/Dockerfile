# YOLOv5 ROS - HSR互換環境
# Based on tidbots_docker/hsr-tid/docker-yolov5-ros

FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir
RUN rm -rf /opt/conda

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user

ENV SHELL /bin/bash
ENV DEBCONF_NOWARNINGS=yes
ENV DEBIAN_FRONTEND noninteractive

#-----------------------------
# install common library
#-----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python-is-python3 \
    git \
    sudo \
    nano\
    curl \
    screen \
    g++ \
    wget \
    net-tools \
    software-properties-common \
    make \
    libopencv-dev

#-----------------------------
# install additional packages
#-----------------------------
RUN pip3 install --upgrade pip
RUN pip3 install setuptools \
    opencv-python \
    numpy \
    rospkg


# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext torch torchvision
RUN pip install --no-cache -r requirements.txt albumentations comet gsutil notebook Pillow>=9.1.0 \
    'opencv-python<4.6.0.66' \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install yolov5

# Set environment variables
ENV OMP_NUM_THREADS=8

#-----------------------------
# intall ros-noetic
#-----------------------------
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get -y update && \
    apt-get -y install ros-noetic-ros-base &&\
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    apt install -y python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep && \
    rosdep init &&\
    rosdep update

RUN apt-get -y install ros-noetic-ros-numpy \
    ros-noetic-smach-ros \
    ros-noetic-cv-bridge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------
# create catkin_ws
#-----------------------------
RUN mkdir -p /catkin_ws/src
RUN cd /catkin_ws/src && /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_init_workspace'

# setup entrypoint
COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

ENV CMAKE_PREFIX_PATH "/catkin_ws/devel:/opt/ros/noetic"

WORKDIR /catkin_ws
