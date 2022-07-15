# Jetson Triton Client

This Dockerfile builds a Triton client image with ROS noetic, CV bridge and OpenCV 4.1 for JetPack L4T 32.5.1.

## Usage

```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t niqbal996/triton-server:client .

docker run -it --rm --runtime nvidia --net=host --name triton-client niqbal996/triton-server:client
```

