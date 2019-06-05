#!/usr/bin/env bash

if [[ "$#" -ne 1 ]]; then
    echo "No evaluation dataset selected.."
    exit 1
fi

if [[ ! -d "$1" ]]; then
  echo "data directory $1 does not exist!"
  exit 1
fi

echo ""
echo ""
echo "execute evaluation on $1"
echo ""
echo ""

GT_DIR="$1/gt"
TR_DIR="$1/tracks3d"

if [[ ! -d "$GT_DIR" ]]; then
    echo "No ground truth found!"
    exit 1
fi

if [[ ! -d "$TR_DIR" ]]; then
    echo "No 3D tracks found! -> $TR_DIR"
    exit 1
fi

nvidia-docker run\
    --privileged\
    --rm\
    -it\
    -v "$PWD":/home/user/mv3dpose:ro\
    -v "$1":/home/user/dataset\
    jutanke/mv3dpose\
    /bin/bash exec_eval.sh