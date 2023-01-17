ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.04-py3
FROM ${BASE_IMAGE} as tritonserver

USER root
# Installing boost and spconv 
RUN mkdir -p /opt/dependencies 
COPY ./docker/server_3d/ /opt/dependencies
WORKDIR /opt/dependencies
RUN /opt/dependencies/install_spconv.sh

# install Det3d 
RUN python3 -m pip install det3d-1.0rc0+01258d8-cp38-cp38-linux_x86_64.whl
WORKDIR /opt/tritonserver/