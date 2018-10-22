"""
Purpose:
Iterate through all images and record image metadata.
Primarily: the image size so that fractional features which are recorded by annotators
can be applied to either the original image or the scaled down 1024-max dimension image.
"""
import os
import sys
import csv
import time
from PIL import Image
from tqdm import tqdm

from annotate.data import get_img_fn, get_metadata_file
from annotate.utils import line_count


fn = get_metadata_file("train-images-boxable-with-rotation.csv")


max_count = 10000
if max_count is not None:
    total_lines = max_count
else:
    total_lines = line_count(fn)
count = 0

with open(fn,'r') as csvfile:
    reader = csv.reader(csvfile)
    start_t = time.time()
    for row in tqdm(reader, total=total_lines):
        if count == 0:
            count+=1
            continue
        row_id = row[0]
        rotation = row[-1]
        if rotation:
            rotation = float(rotation)
        else:
            rotation = 0.0
        img_path = get_img_fn(row_id)
        im = Image.open(img_path)
        width, height = im.size
        sys.stdout.write("%s %d %d %.1f\n" %(row_id, width, height, rotation))
        count += 1
        if max_count is not None and count > max_count:
            break
    end_t = time.time()
    print(max_count / float(end_t - start_t), " it/second")
    sys.exit(0)