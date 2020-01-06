FROM tensorflow/tensorflow:2.0.0-py3-jupyter

USER root
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension


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

# --- <Populate jupyter> ---
ENV SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/$NB_USER

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

EXPOSE 8888

USER $NB_USER
WORKDIR $HOME/work
ENTRYPOINT xvfb-run -s "-screen 0 1400x900x24" jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root --LabApp.token=''