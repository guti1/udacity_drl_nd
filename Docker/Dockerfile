FROM tensorflow/tensorflow:2.0.0-py3-jupyter

USER root
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y  \
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    freeglut3 \
    python-opengl \
    libboost-all-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libsdl2-2.0-0\
    libgles2-mesa-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    xserver-xorg-input-void \
    xserver-xorg-video-dummy \
    python-gtkglext1 \
    xpra \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN jupyter serverextension enable --py jupyterlab --sys-prefix

EXPOSE 8888

WORKDIR /tf
