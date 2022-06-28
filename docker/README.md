# Jetson Triton Client

This Dockerfile builds a Triton client image with ROS noetic, CV bridge and OpenCV 4.1 for JetPack L4T 32.5.1.

## Usage

```bash
sudo -H DOCKER_BUILDKIT=1 nvidia-docker build --build-arg WHEEL_FILE=docker/jetson/onnxruntime_gpu-1.8.0-cp36-cp36m-linux_aarch64.whl -f docker/jetson/Dockerfile -t ag-infer-client:latest .

docker run -it --rm --runtime nvidia --net=host --name client-test ag-infer-client
```

