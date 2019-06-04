#!/usr/bin/env bash

echo ""
echo ""
echo "visualize mvpose on $1"
echo ""
echo ""

TRK_DIR="$1/tracks3d"

if [ ! -d "$TRK_DIR" ]; then
  echo "tracks directory $TRK_DIR does not exist!"
  exit 1
fi

nvidia-docker run\
    --privileged\
    --name='mv3dpose_vis'\
    --rm\
    -it\
    -v "$PWD":/home/user/mv3dpose:ro\
    -v "$1":/home/user/dataset\
    jutanke/mv3dpose\
    /bin/bash exec_vis.sh