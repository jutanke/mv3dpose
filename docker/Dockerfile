FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN echo "0.0.1"

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN apt-get install -y build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    gfortran \
    libjpeg8-dev libpng-dev \
    libjasper1 \
    libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
    libxine2-dev libv4l-dev \
    vim htop tmux wget curl

RUN cd /usr/include/linux && ln -s -f ../libv4l1-videodev.h videodev.h

RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev libtbb-dev qt5-default \
    libatlas-base-dev \
    libfaac-dev libmp3lame-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libavresample-dev \
    x264 v4l-utils \
    libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev \
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
    mesa-utils and libgl1-mesa-glx x11-apps eog \
    liblmdb-dev libleveldb-dev libsnappy-dev &&\
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7 &&\
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

#RUN conda install python==3.6?
RUN conda install -y numpy \
    conda \
    anaconda-client \
    conda-build \
    curl \
    cudatoolkit\
    greenlet \
    cython \
    fastcache \
    flask \
    graphviz \
    h5py \
    hdf5 \
    jpeg \
    jsonschema \
    jupyter \
    jupyter_client \
    jupyter_console \
    jupyter_core \
    libffi \
    libsodium \
    libpng \
    libtiff \
    libxml2 \
    libxslt \
    llvmlite \
    lxml \
    matplotlib \
    mistune \
    notebook \
    nltk \
    nose \
    numba \
    openblas \
    pandas \
    pep8 \
    pylint \
    pycurl \
    pycosat \
    psutil \
    ptyprocess \
    scikit-image \
    scikit-learn \
    scipy \
    tqdm

RUN conda update conda && conda update --all

RUN apt-get install -y python3-dev libgoogle-glog-dev libgflags-dev libprotobuf-dev protobuf-compiler\
    libgoogle-glog-dev libgflags-dev\
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen\
    x264 v4l-utils libavresample-dev\
    libopencore-amrnb-dev libopencore-amrwb-dev\
    libvorbis-dev libxvidcore-dev\
    libfaac-dev libmp3lame-dev libtheora-dev\
    libatlas-base-dev libgtk2.0-dev libtbb-dev qt5-default\
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

RUN apt-get install -y libxine2-dev libv4l-dev\
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev\
    libtiff-dev libjasper1 libjpeg8-dev libpng-dev\
    git gfortran build-essential checkinstall cmake pkg-config yasm

RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.2.0
RUN cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D ENABLE_FAST_MATH=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D BUILD_EXAMPLES=ON ..

RUN cd opencv && cd build && make -j20 && make install

RUN echo "0.0.2"
COPY exec.sh /exec.sh
COPY exec_vis.sh /exec_vis.sh
COPY exec_eval.sh /exec_eval.sh
COPY exec_test_perms.sh /exec_test_perms.sh
