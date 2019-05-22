#!/usr/bin/env bash

echo "execute mvpose on $1"

VID_DIR="$1/videos"
CAM_DIR="$1/cameras"
POSE_DIR="$1/poses"
OPENPOSE_DIR="$PWD/openpose"

echo "$OPENPOSE_DIR"

if [ ! -d "$VID_DIR" ]; then
  echo "video directory $VID_DIR does not exist!"
  exit 1
fi

N_CAMS=$(ls -l $VID_DIR | grep -c ^d)

echo "#cameras: $N_CAMS"

if [ ! -d "$POSE_DIR" ]; then
    mkdir $POSE_DIR
    echo "execute 2D pose estimation..."
    for ((CID=0; CID<$N_CAMS; CID++))
    do
        echo -e "\t 2D pose estimation for camera $CID"
        CAMERA=$(printf "camera%02d" $CID)
        POSE2D_INPUT="$VID_DIR/$CAMERA"
        POSE2D_OUTPUT="$POSE_DIR/$CAMERA"
        mkdir $POSE2D_OUTPUT
        cd $OPENPOSE_DIR && ./openpose.sh $POSE2D_INPUT $POSE2D_OUTPUT
    done
else
    echo "2D poses already estimated"
fi

nvidia-docker run\
    --privileged\
    --name='mv3dpose_exec'\
    --rm\
    -it\
    -v "$PWD":/home/user/mv3dpose:ro\
    -v "$1":/home/user/dataset\
    jutanke/mv3dpose\
    /bin/bash exec.sh