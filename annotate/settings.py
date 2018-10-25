import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))

# Open Images is large, so we partition by the first character of the id
# which makes it easier to spread the data across volumes.
# This is also how Open Images is distributed via archives.
OPEN_IMAGES_DIR_MAP = {
 "meta": "/mnt/Apps3/datasets/open-images",
 "test": "/mnt/Apps3/datasets/open-images",
 "validation": "/mnt/Apps3/datasets/open-images",
 "0": "/mnt/nvm-data/open-images",
 "1": "/mnt/nvm-data/open-images",
 "2": "/mnt/nvm-data/open-images",
 "3": "/mnt/nvm-data/open-images",
 "4": "/mnt/nvm-data/open-images",
 "5": "/mnt/nvm-data/open-images",
 "6": "/mnt/nvm-data/open-images",
 "7": "/mnt/nvm-data/open-images",
 "8": "/mnt/nvm-data/open-images",
 "9": "/mnt/nvm-data/open-images",
 "a": "/mnt/ssd-data/open-images",
 "b": "/mnt/ssd-data/open-images",
 "c": "/mnt/ssd-data/open-images",
 "d": "/mnt/ssd-data/open-images",
 "e": "/mnt/ssd-data/open-images",
 "f": "/mnt/ssd-data/open-images",
}

# Setting a single directory overrides the mapping
OPEN_IMAGES_DIR = None
if OPEN_IMAGES_DIR:
    for k in OPEN_IMAGES_DIR_MAP.keys():
        OPEN_IMAGES_DIR_MAP[k] = OPEN_IMAGES_DIR

ANNOTATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
LIB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib"))