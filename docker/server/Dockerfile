### DOWNLOAD MODEL STEP
FROM python:3.9.12-slim as model_provisioning

RUN python -m pip install --upgrade pip
RUN pip install python-keycloak==0.27.0 boto3==1.18.0
COPY utils/download_model_s3_keycloak.py .


RUN python download_model_s3_keycloak.py \
    --username "naeem.iqbal@dfki.de" \
    --password "hubble123" \
    --minio-endpoint-url "https://data-sharing.platform.agri-gaia.com" \
    --keycloak-endpoint-url "https://account.platform.agri-gaia.com/auth/" \
    --keycloak-client-id "minio" \
    --keycloak-realm-name "Data-Sharing" \
    --object-path "niqbal/models/yolov5ncrop.onnx" \
    --output-path "/tmp/weed_detector"


### DOWNLOAD TRITON STEP
FROM alpine as triton_download

RUN wget --no-check-certificate -nv https://github.com/triton-inference-server/server/releases/download/v2.19.0/tritonserver2.19.0-jetpack4.6.1.tgz
RUN mkdir triton && \
    tar -xzf tritonserver2.19.0-jetpack4.6.1.tgz -C triton

# ADD TRITON STEP

FROM nvcr.io/nvidia/l4t-base:r32.6.1


ARG UNAME=docker
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd --create-home -u $UID -g $GID -o --shell /bin/bash $UNAME && echo "$UNAME:$UNAME" | chpasswd && adduser $UNAME sudo && usermod -a -G video $UNAME && usermod -a -G dialout $UNAME
USER $UNAME
WORKDIR /home/$UNAME
USER root
ENV DEBIAN_FRONTEND=noninteractive

######################################
##### Install triton dependicies #####
######################################


RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            software-properties-common \
            autoconf \
            automake \
            build-essential \
            git \
            libb64-dev \
            libre2-dev \
            libssl-dev \
            libtool \
            libboost-dev \
            rapidjson-dev \
            patchelf \
            pkg-config \
            libopenblas-dev \
            libarchive-dev \
            zlib1g-dev \
            python3 \
            python3-pip \
            python3-dev \
            wget \
            curl


# Install Pytorch dependencies
RUN apt-get -y install autoconf \
            bc \
            g++-8 \
            gcc-8 \
            clang-8 \
            lld-8
# Upgrade pip to latest version first to avoid unnecessary setup.py builds for python modules
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools

RUN python3 -m pip install --upgrade expecttest xmlrunner hypothesis aiohttp pyyaml scipy ninja typing_extensions protobuf

# Install pytorch
RUN python3 -m pip install --upgrade torch

#upgrade cmake to 3.21
RUN apt remove cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
      gpg --dearmor - | \
      tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake-data=3.21.0-0kitware1ubuntu18.04.1 cmake=3.21.0-0kitware1ubuntu18.04.1

# install triton python dependencies
RUN apt-get install -y --no-install-recommends \
            curl \
            jq
RUN python3 -m pip install --upgrade wheel setuptools cython && \
    python3 -m pip install --upgrade grpcio-tools numpy==1.19.4 future attrdict && \
    python3 -m pip install --upgrade six requests flake8 flatbuffers pillow

######################################
###### Install triton server #########
######################################

# Download tritonserver binaries
RUN mkdir -p /home/docker/tritonserver && \
    mkdir -p /model_repository && \
    chown $UNAME /model_repository


WORKDIR /home/docker/tritonserver

COPY --from=triton_download triton ./


USER $UNAME
ENV PATH=$PATH:/home/docker/tritonserver/bin
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/llvm-8/lib"

# ENV OPENBLAS_CORETYPE=ARMV8 this is not needed with numpy 1.19.4
CMD ["tritonserver", "--model-repository=/model_repository", "--backend-directory=/home/docker/tritonserver/backends", "--backend-config=tensorflow,version=2"]

### ADD MODEL STEP


RUN mkdir -p "/model_repository/weed_detector/1"
COPY --from=model_provisioning "/tmp/weed_detector" "/model_repository/weed_detector/1/model.onnx"
COPY "models/weed_detector/config.pbtxt" "/model_repository/weed_detector/config.pbtxt"
