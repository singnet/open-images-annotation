#!/bin/bash

# This currently doesn't download the annotations themselves
if [ -z ${OPEN_IMAGES_DIR+x} ]
then
    echo "You need to set the env var OPEN_IMAGES_DIR";
    exit 1
fi

cd $OPEN_IMAGES_DIR

mkdir -p archives
for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f
do
   aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_$i.tar.gz archives/
done
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz archives/
aws s3 --no-sign-request cp s3://open-images-dataset/tar/test.tar.gz archives/
aws s3 --no-sign-request cp s3://open-images-dataset/tar/challenge2018.tar.gz archives/