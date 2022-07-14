#!/usr/bin/env bash
mkdir -p /catkin_ws/src
cd /catkin_ws/src/
wget --no-check-certificate https://github.com/niqbal996/vision_opencv/archive/refs/heads/noetic.zip
apt-get update && apt-get install unzip
unzip noetic.zip "vision_opencv-noetic/cv_bridge/*" -d "./"
mv vision_opencv-noetic/cv_bridge  ./
rm -rf vision_opencv-noetic
cd /catkin_ws/
source /opt/ros/noetic/setup.bash
catkin_make --only-pkg-with-deps cv_bridge
