#!/usr/bin/env bash

docker run\
    --gpus all\
    --privileged\
    --name='mv3dpose_run'\
    --rm\
    -it\
    jutanke/mv3dpose\
    /bin/bash