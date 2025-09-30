#!/bin/bash

if [ $# -eq 1 ];
then
    container_name=$1
else
    container_name="nlp-ctr"
fi

if [ $# -eq 2 ];
then
    docker_name=$1
else
    docker_name="nlp"
fi
dir_up=$(realpath "../../")
docker run -v $dir_up:/scripts/ -it --rm --name $container_name $docker_name bash