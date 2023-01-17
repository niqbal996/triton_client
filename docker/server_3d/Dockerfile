ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.04-py3
FROM ${BASE_IMAGE} as tritonserver

USER root
# Installing boost and spconv 
RUN mkdir -p /opt/dependencies 
COPY ./docker/server_3d/ /opt/dependencies
WORKDIR /opt/dependencies
RUN /opt/dependencies/install_spconv.sh

# install Det3d 
RUN git clone https://github.com/poodarchu/Det3D.git && \
    cp requirements_det.txt ./Det3D/requirements.txt && \ 
    cd Det3D && python3 setup.py build develop
