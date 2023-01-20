ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.04-py3
FROM ${BASE_IMAGE} as tritonserver

USER root
# Installing boost and spconv 
RUN mkdir -p /opt/dependencies 
COPY ./docker/server_3d/ /opt/dependencies
WORKDIR /opt/dependencies
RUN /opt/dependencies/install_spconv.sh

# install Det3d 1.0 
# RUN python3 -m pip install det3d-1.0rc0+01258d8-cp38-cp38-linux_x86_64.whl
# install OpenPCDet 0.6 
RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && \
    python3 setup.py develop && \
    echo 'export PYTHONPATH=$PYTHONPATH:/opt/dependencies/OpenPCDet' >> ~/.bashrc && \
    python3 -m pip install numpy==1.23.5

WORKDIR /opt/tritonserver/