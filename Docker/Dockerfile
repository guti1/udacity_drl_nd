FROM tensorflow/tensorflow:2.0.0-py3-jupyter

USER root
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl\
       x11-utils \
       g++  \
       git  \
       cmake \
       zlib1g-dev \
       libjpeg-dev \
       libboost-all-dev \
       libsdl2-dev \
       swig \
       python3  \
       python3-dev  \
       python3-future  \
       python3-pip  \
       python3-setuptools  \
       python3-wheel  \
       python3-tk \
       libopenblas-base  \
       libatlas-base-dev


COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV DEBIAN_FRONTEND teletype
ENV DISPLAY=:0
EXPOSE 8888

WORKDIR /tf
