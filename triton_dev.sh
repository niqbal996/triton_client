apt install ros-noetic-cv-bridge ros-noetic-vision-msgs
apt install python3-pip
python3 -m pip install tritonclient[all] 
python3 -m pip install torch==1.8.1 torchvision==0.9.1
export PYTHONPATH=$PYTHONPATH:/opt/triton_client/seerep