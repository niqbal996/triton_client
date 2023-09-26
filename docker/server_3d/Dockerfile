ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.12-py3
FROM ${BASE_IMAGE} as tritonserver

USER root
# Installing boost and spconv 
RUN mkdir -p /opt/dependencies 
COPY ./docker/server_3d/ /opt/dependencies
WORKDIR /opt/dependencies
RUN --mount=type=cache,target=/root/.cache /opt/dependencies/install_spconv.sh

# Installing torch 1.13.1 compiled with CUDA 11.7
RUN --mount=type=cache,target=/root/.cache /opt/dependencies/install_torch_2.sh

# install OpenPCDet 0.6.0  
# RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
#     cd OpenPCDet && \
#     python3 setup.py develop && \ # python3 -m pip install -e .
#     echo 'export PYTHONPATH=$PYTHONPATH:/opt/dependencies/OpenPCDet' >> ~/.bashrc && \
#     python3 -m pip install numpy==1.23.5 && \ 
#     python3 -m pip install kornia==0.6.8

WORKDIR /opt/tritonserver/