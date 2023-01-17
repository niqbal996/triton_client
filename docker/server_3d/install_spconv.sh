#!/usr/bin/env bash
python3 -m pip install --upgrade pip
apt-get update && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
dpkg -i cuda-keyring_1.0-1_all.deb && \ # download and install latest cuda keyrings
apt-key del 7fa2af80 && \ # remove outdated keys
apt-get install -y libboost-all-dev && \ # install boost
pip install -r requirements.txt # install python dependencies 