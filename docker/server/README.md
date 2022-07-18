# Triton Server Container

Instructions to build the container from the repos root directory:

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/arm64 build -f docker/server/Dockerfile -t <image tag>
```

To run the container:

```bash
docker run --runtime nvidia --name triton-server --user docker -p 8000:8000 -p 8001:8001 -p 8002:8002 <image tag>
```


The ports are to for the following endpoints:
- 8000: KServe v2 REST endpoint
- 8001: KServe v2 GRPC endpoint
- 8002: Endpoint for Metrics Service

Additional documentation for the API can be found here

- https://github.com/kserve/kserve/tree/master/docs/predict-api/v2
- https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md