# Run example detections on pointclouds from AITF carrier
At host system:
```bash
export DISPLAY=:1
sudo xhost + #so that docker can connect to DISPLAY:1 
docker pull djiajun1206/pcdet # fetch the image
docker run -it --runtime=nvidia \
--net=host \
--volume=/home/naeem/.Xauthority:/root/.Xauthority:rw \
----env=DISPLAY \
djiajun1206/pcdet
```

Once you go inside the docker container:
```bash
nvidia-smi
Tue Aug  2 15:25:06 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.48.07    Driver Version: 515.48.07    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   50C    P8    11W /  N/A |   1196MiB /  8192MiB |     12%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```
Based on the NVIDIA driver version **515.xx.xx**, install corresponding OpenGL version. 
```bash
apt search libnvidia-gl-515
Sorting... Done
Full Text Search... Done
libnvidia-gl-430/unknown 515.48.07-0ubuntu1 amd64
  Transitional package for libnvidia-gl-515

libnvidia-gl-515/unknown,now 515.48.07-0ubuntu1 amd64 [installed]
  NVIDIA OpenGL/GLX/EGL/GLES GLVND libraries and Vulkan ICD

libnvidia-gl-515-server/bionic-updates,bionic-security 515.48.07-0ubuntu0.18.04.1 amd64
  NVIDIA OpenGL/GLX/EGL/GLES GLVND libraries and Vulkan ICD
apt install libnvidia-gl-515
```
Once inside the container:
```bash
cd /workspace/OpenPCDet-master
python demo.py --cfg_file /workspace/OpenPCDet-master/tools/cfgs/kitti_models/second.yaml \ 
--ckpt /workspace/second_7862.pth \
--data_path /workspace/bag_data_pc \
--ext .npy
```
The visualization window from Open3D visualizer should pop up on your desktop screen outside the docker environment. 