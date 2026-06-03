#!/bin/bash

echo "Building Docker image..."

# Build the image from the Dockerfile in the current directory
docker build -t fatrop_generalized_dynamics_image .

# clear
echo "Starting docker container..."

# Start docker container
xhost +local:docker

docker run -it \
    -v .:/generalized_dynamics_examples \
    --name fatrop_generalized_dynamics_container \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --net=host \
    --privileged \
    fatrop_generalized_dynamics_image \
    bash
