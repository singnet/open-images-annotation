#!/bin/bash
set -e 
set -o pipefail

if [ -z ${OPEN_IMAGES_DIR+x} ]
then
    echo "You need to set the env var OPEN_IMAGES_DIR";
    exit 1
fi
if [ -z ${OPEN_IMAGES_ARCHIVE_DIR+x} ]
then
    echo "You need to set the env var OPEN_IMAGES_ARCHIVE_DIR";
    exit 1
fi

cd $OPEN_IMAGES_DIR

for i in archives/train_*.tar.gz
do
    echo "Extracting $i"
    pv -p --time --eta --rate $i | tar xzf - -C train/
    echo "Moving $i to archive destination $OPEN_IMAGES_ARCHIVE_DIR"
    rsync -P $i $OPEN_IMAGES_ARCHIVE_DIR
    rm $i
done