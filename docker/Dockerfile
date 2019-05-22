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
    vim htop tmux wget

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
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

RUN apt-get install -y liblmdb-dev libleveldb-dev libsnappy-dev

RUN apt-get install -y mesa-utils and libgl1-mesa-glx x11-apps eog curl
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

RUN conda install python==3.6
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
    jinja2 \
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
    pyqt \
    pycurl \
    pycosat \
    psutil \
    ptyprocess \
    redis \
    redis-py \
    scikit-image \
    scikit-learn \
    scipy \
    simplegeneric \
    sip \
    sockjs-tornado \
    sqlite \
    sqlalchemy \
    tqdm \
    werkzeug \
    unicodecsv \
    zeromq \
    zlib

# RUN pip install tensorflow-gpu==2.0.0-alpha0
RUN pip install tensorflow-gpu

RUN conda install -y -c menpo opencv3

RUN conda update conda & conda update --all

RUN pip install transforms3d
RUN wget -r -l1 -np -nd -nc http://cdaweb.gsfc.nasa.gov/pub/software/cdf/dist/latest-release/linux/ -A cdf*-dist-all.tar.gz &&\
    tar xf cdf*-dist-all.tar.gz -C ./ && \
    cd cdf*dist &&\
    apt install -y build-essential gfortran libncurses5-dev && \
    make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all &&\
    make install #no sudo
RUN pip install git+https://github.com/spacepy/spacepy.git

RUN echo 'export CDF_BASE=/cdf37_0-dist' >> ~/.bashrc &&\
    echo 'export CDF_INC=$CDF_BASE/include' >> ~/.bashrc &&\
    echo 'export CDF_LIB=$CDF_BASE/lib' >> ~/.bashrc &&\
    echo 'export CDF_BIN=$CDF_BASE/bin' >> ~/.bashrc &&\
    echo 'export LD_LIBRARY_PATH=$CDF_BASE/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

ENV CDF_BASE /cdf37_0-dist
ENV CDF_INC /cdf37_0-dist/include
ENV CDF_LIB /cdf37_0-dist/lib
ENV CDF_BIN /cdf37_0-dist/bin