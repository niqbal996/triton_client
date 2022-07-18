ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:21.10-py3
FROM ${BASE_IMAGE} as onnxruntime

#ARG WHEEL_FILE
ARG UNAME=docker
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd --create-home -u $UID -g $GID -o --shell /bin/bash $UNAME && echo "$UNAME:$UNAME" | chpasswd && adduser $UNAME sudo && usermod -a -G video $UNAME && usermod -a -G dialout $UNAME
USER $UNAME
WORKDIR /home/$UNAME
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Installing ROS noetic..."
ARG ROS_PKG=ros_base
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
        build-essential \
        software-properties-common \
        python3-pip \
        python3-dev \
        cmake \
        unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/NVIDIA/nvidia-docker/issues/1632
#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y lsb-release curl rapidjson-dev && apt-get clean all
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends ros-noetic-ros-base && \
    source /opt/ros/noetic/setup.bash && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    apt install --no-install-recommends -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    apt install -y ros-noetic-image-transport ros-noetic-cv-bridge ros-noetic-vision-msgs && \
    mkdir -p /catkin_ws/src && \
    cd /catkin_ws && \
    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 && \
    echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    rosdep init && rosdep update --rosdistro $ROS_DISTRO

#RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src/
#RUN /workspace/triton_client/docker/install_cvbridge.sh
#RUN echo 'source /catkin_ws/devel/setup.bash' >> ~/.bashrc
# download/build the ROS source
#RUN mkdir ~/catkin_ws && \
#    cd ~/catkin_ws && \
#    rosinstall_generator ${ROS_PKG} vision_msgs --rosdistro ${ROS_DISTRO} --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
#    mkdir src && \
#    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
#    apt-get update && \
#    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
#    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
#    rm -rf /var/lib/apt/lists/*

# install python libraries
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tritonclient[grpc] --ignore-installed pyyaml~=6.0 numpy==1.19.4
RUN python3 -m pip install torch==1.8.1 torchvision==0.9.1

# install PCDet
RUN apt-get install unzip
#RUN git clone git@github.com:open-mmlab/OpenPCDet.git && git checkout 846cf3e && cd OpenPCDet && python setup.py develop

#RUN wget --no-check-certificate https://github.com/open-mmlab/OpenPCDet/archive/refs/heads/master.zip &&  \
##    mkdir -p /workspace/pcdet && \
#    unzip master.zip -d /workspace/ && \
#    cd /workspace/OpenPCDet-master && python3 setup.py develop

#USER $UNAME
RUN mkdir -p /workspace/triton_client
COPY . /workspace/triton_client/
WORKDIR /workspace/triton_client/
#RUN cd utils/export_models/OpenPCDet && python setup.py develop
#CMD ["python3", "-u", "main.py", "weed_detector"]
