#!/usr/bin/env bash

nvidia-docker run\
    --privileged\
    --name='mv3dpose_run'\
    --rm\
    -it\
    jutanke/mv3dpose\
    /bin/bash