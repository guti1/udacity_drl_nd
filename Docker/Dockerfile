FROM tensorflow/tensorflow:1.7.1-py3

USER root
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

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

RUN cp /run_jupyter.sh /usr/local/bin/start-notebook.sh

USER $NB_USER
WORKDIR $HOME/work