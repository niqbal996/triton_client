#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash 
python3 -u /workspace/triton_client/main.py -m weed_detector
