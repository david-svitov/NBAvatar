#!/usr/bin/env bash
PORT=8088

PARAMS="-p ${PORT}:${PORT} --net=host --ipc=host -u $(id -u ${USER}):$(id -g ${USER})"
NAME="textured-avatar"
VOLUMES="-v /:/mounted"
