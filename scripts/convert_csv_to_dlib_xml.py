import csv
import argparse
import os
import sys
import cv2
import dlib
from skimage import io
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from annotate.data import get_img_fn


def convert_positions_to_absolute(img, row_f):
    width, height = img.size
    result = []
    for i in range(0, len(row_f) // 2):
        result.append(int(row_f[i*2] * width))
        result.append(int(row_f[i*2 + 1] * height))
    if len(result) < len(row_f):
        raise Exception("Odd number of columns, not sure how to convert to absolute")
    return result


def line_count(fn):
    return sum((1 for i in open(fn, 'rb')))


def debug_boxes(win, img_path, dims):
    cvimg = io.imread(img_path)
    if len(cvimg.shape) == 2:
        rgb_img = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2RGB)
    else:
        rgb_img = cvimg
    cv2.rectangle(rgb_img, (dims[0], dims[1]), (dims[2], dims[3]), (0, 255, 0), 2)
    # cv2.rectangle(rgb_img, (dims[4], dims[5]), (dims[6], dims[7]), (0, 0, 255), 2)

    win.set_image(rgb_img)
    dlib.hit_enter_to_continue()


def convert_file(fn, max_count=None, test_count=10000, source_filter="face_bbox", boxes_only=False, square_boxes=False):
    debug = False
    out_fn = "training.xml"
    test_fn = "testing.xml"
    while os.path.exists(out_fn):
        out_fn = out_fn.split(".")[0] + "_.xml"
    while os.path.exists(test_fn):
        test_fn = test_fn.split(".")[0] + "_.xml"
    print("Writing output to:", out_fn, test_fn)

    if max_count is not None:
        max_count = max(max_count, test_count+1)
    total_lines = max_count if max_count is not None else line_count(fn)
    count = 0
    oddities = 0
    last_image = None
    with open(out_fn,'w') as trainfile, open(test_fn,'w') as testfile, open(fn,'r') as f:
        trainfile.write(
"""<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>Training faces</name>
<comment>These are images from open images.</comment>
<images>
""")
        testfile.write(
"""<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>Testing faces</name>
<comment>These are images from open images.</comment>
<images>
""")
        reader = csv.reader(f)
        outfile = testfile

        win = None
        if debug:
            win = dlib.image_window()

        for row in tqdm(reader, total=total_lines):
            row_id = row[0]
            if count == 0:
                count += 1
                continue
            img_path = get_img_fn(row_id)

            if outfile == testfile and count > test_count and row_id != last_image:
                outfile = trainfile
                last_image = None
                testfile.write(" </image>\n</images>\n</dataset>\n")

            if row_id != last_image:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    if len(img.size) > 2 or max(img.size) > 1024:
                        oddities += 1
                        img = None
                    else:
                        if last_image is not None:
                            outfile.write(" </image>\n")
                        outfile.write(" <image file='%s'>\n" % img_path)
                        last_image = row[0]
                else:
                    img = None
                
            if img is not None:
                if len(row) > 5:
                    dims = convert_positions_to_absolute(img, [float(r) for r in row[2:]])
                    # Output original OI boxes if specifically requested, otherwise generated bounding boxes
                    if source_filter != 'bbox' and max(dims[4:8]) > 0:
                        a_box = write_box_and_landmarks(outfile, img, dims[4:8], dims[8:], boxes_only, square_boxes)
                    elif max(dims[0:4]) > 0:
                        a_box = write_box_and_landmarks(outfile, img, dims[0:4], dims[8:], boxes_only, square_boxes)
                else:
                    dims = convert_positions_to_absolute(img, [float(r) for r in row[1:]])
                    # This is just a csv with bounding boxes
                    a_box = write_box_and_landmarks(outfile, img, dims[0:4], [], True, square_boxes)
                count += 1

                if debug:
                    debug_boxes(win, img_path, a_box)

            if max_count is not None and count > max_count:
                trainfile.write(" </image>\n</images>\n</dataset>\n")
                return
        trainfile.write(" </image>\n</images>\n</dataset>\n")
        testfile.write(" </image>\n</images>\n</dataset>\n")
        print("image oddities skipped", oddities)


def write_box_and_landmarks(f, img, box_dims, landmarks, boxes_only, square_boxes):
    dims = box_dims
    w = dims[2] - dims[0]
    h = dims[3] - dims[1]
    if square_boxes:
        iw, ih = img.size
        size = max(w, h)
        half_size = size // 2
        centre_x = dims[0] + (w // 2)
        centre_y = dims[1] + (h // 2)
        dims = (max(0, centre_x - half_size), max(0, centre_y - half_size)) #, min(iw, centre_x + half_size), min(ih, centre_y + half_size))
        w = size
        h = size

    if boxes_only:
        f.write("  <box top='%d' left='%d' width='%d' height='%d'/>\n" % (dims[1], dims[0], w, h))
        return

    f.write("  <box top='%d' left='%d' width='%d' height='%d'>\n" % (dims[1], dims[0], w, h))
    for i in range(len(landmarks) // 2):
        f.write("    <part name='%02d' x='%d' y='%d'/>\n" % (i, landmarks[i*2], landmarks[i*2+1]))
    f.write("  </box>\n")
    return (dims[0], dims[1], dims[0]+w, dims[1]+h)


def main():
    parser = argparse.ArgumentParser(description="Take a face landmarks csv and convert it to xml that dlib understands.")
    parser.add_argument("csv",
        help="CSV file containing landmarks"
    )
    parser.add_argument("--open-images",
        help="Use the open images bounding box",
        required=False, action="store_true"
    )
    parser.add_argument("--boxes-only",
        help="Only output bounding boxes to xml",
        required=False, action="store_true"
    )
    parser.add_argument("--square-boxes",
        help="Make boxes square",
        required=False, action="store_true"
    )
    parser.add_argument("--max-count",
        help="Only process this many lines, good for testing a subset of the data",
        required=False, action="store", type=int, default=None
    )
    parser.add_argument("--test-count",
        help="Take the first N rows and use these as testing.xml",
        required=False, action="store", type=int, default=500
    )
    args = parser.parse_args()

    source_filter = None
    if args.open_images:
        source_filter = "bbox"

    convert_file(args.csv, max_count=args.max_count, test_count=args.test_count, source_filter=source_filter, boxes_only=args.boxes_only, square_boxes=args.square_boxes)


if __name__ == "__main__":
    main()
