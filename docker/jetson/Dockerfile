ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r32.5.0-py3
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

# add the ROS deb repo to the apt sources list
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
        libopenblas-dev \
        libpython3.6-dev \
        python3-pip \
        python3-dev \
        cmake \
        unattended-upgrades \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# install bootstrap dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libpython3-dev \
          python3-rosdep \
          python3-rosinstall-generator \
          python3-vcstool \
          build-essential && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*

# download/build the ROS source
RUN mkdir ~/catkin_ws && \
    cd ~/catkin_ws && \
    rosinstall_generator ${ROS_PKG} vision_msgs --rosdistro ${ROS_DISTRO} --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
    mkdir src && \
    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
    apt-get update && \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*
#USER $UNAME
SHELL ["/bin/bash", "-c"] 
RUN echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN mkdir -p /workspace/triton_client
COPY . /workspace/triton_client/
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tritonclient[grpc] --ignore-installed pyyaml~=6.0 numpy==1.19.4
RUN /workspace/triton_client/docker/install_cvbridge.sh
RUN echo 'source /catkin_ws/devel/setup.bash' >> ~/.bashrc
WORKDIR /workspace/triton_client/
#CMD ["/bin/bash", "python3", "main.py", "-m", "weed_detector"]
ENTRYPOINT /workspace/triton_client/docker/infer_client.sh
