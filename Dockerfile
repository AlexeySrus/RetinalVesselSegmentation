# To use CUDA please uncomment the following line:
# FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
# And comment the following one:
FROM ubuntu:20.04

EXPOSE 9009

# configure timezone, our app depends on it.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
# RUN /usr/bin/ln -sf /usr/share/zoneinfo/America/Toronto /etc/localtime


RUN apt-get update && \
    apt-get install --no-install-recommends -y curl screen python3 build-essential python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y && apt clean

COPY . /app
WORKDIR /app/
ENV PYTHONPATH=/app

RUN pip3 install -r requirements.txt

RUN mkdir data/
RUN pip3 install gdown
WORKDIR /app/data/
RUN gdown "https://drive.google.com/uc?id=1BpmQegcb06zc4HnmIrZSpK3yaVTf5GRK"
WORKDIR /app/

CMD python3 flask_service.py