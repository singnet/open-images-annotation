"""
browse and inspect open-images data and added annotations
"""
from skimage import io
import numpy as np
import os
import random
import cv2
import dlib
from tqdm import tqdm

from annotate.data import check_all_files_exist, load_hierarchy, summarise_class, get_img_fn, OpenImagesBoxAnnotator
from annotate.faces import FaceBoundingBoxAnnotator, FaceLandmarkAnnotator
from annotate import settings

from skimage.transform import rescale, resize, downscale_local_mean


def normalise_check(img):
    if len(img.shape) == 1 and img.shape[0] >= 2:
        # Some multi frame jpg files cause issues
        # https://github.com/scikit-image/scikit-image/issues/2406
        # https://github.com/python-pillow/Pillow/blob/master/src/PIL/MpoImagePlugin.py
        img = img[0]
    if max(img.shape) > 1024:
        # multiframe jpg files have not been resized correctly
        factor = 1024.0 / max(img.shape)
        img = (rescale(img, factor, anti_aliasing=True, multichannel=True, mode='reflect') * 255).astype(np.uint8)
    return img


def binary_search(lst, value):
    count = len(lst)
    first = 0
    while count:
        it = first
        step = count // 2
        it += step
        if value == lst[it]:
            return it 
        elif value > lst[it]:
            it += 1
            first = it
            count -= step + 1
        else:
            count = step
    return -1


def bulk_annotate(image_ids, to_skip, continue_from=None):
    image_ids_list = list(image_ids.keys())
    image_ids_list.sort()
    total = len(image_ids_list)

    annotators = [
        OpenImagesBoxAnnotator(),
        FaceBoundingBoxAnnotator('hog'),
        FaceLandmarkAnnotator(),
    ]

    start_id_idx = 0
    if continue_from:
        start_id_idx = None
        print("Attempting to continue from ", continue_from)
        for a in annotators:
            versions = a.get_versions()
            # TODO: use a better way to differentiate label only annotators that don't have a model
            if None not in versions:
                continue
            print(versions)
            fn = None
            for v in versions:
                if v is not None and continue_from in ''.join(v.split('.')[0:-1]):
                    print("found version", v)
                    fn = v
            if fn is None:
                err_str = "Couldn't find a matching version for " + str(a.__class__.__name__)
                print(err_str)
                raise Exception(err_str)
            a.load_version(fn)

            idx = binary_search(image_ids_list, a.annotation_set.last_id) + 1
            a.start_from_idx = idx
            if start_id_idx is None or idx < start_id_idx:
                start_id_idx = idx

    if start_id_idx > 0:
        print("Starting from idx", start_id_idx, "which is img_id", image_ids_list[start_id_idx])

    for i, img_id in tqdm(enumerate(image_ids_list[start_id_idx:]), total=total, initial=start_id_idx):
        curr_idx = start_id_idx + i
        if img_id in to_skip:
            continue
        img_path = get_img_fn(img_id)
        #print(img_path)

        # try:
        img = None
        if os.path.exists(img_path):
            img = io.imread(img_path)
            img = normalise_check(img)

            all_annotations = {}
            for a in annotators:
                versions = a.get_versions()
                #print(a.annotator.__class__, version_name, all_annotations)
                if None in versions and curr_idx >= getattr(a, 'start_from_idx', 0):
                    annotations = a.annotate_image(img_id, img, all_annotations)
                else:
                    annotations = a.annotation_set.get_annotation(img_id)
                all_annotations.update(annotations)

                if i % a.flush_interval == 0:
                    a.flush_annotation_set()
        else:
            tqdm.write("No such file %s" % (img_path,))
        # except Exception as e:
        #     import pdb; pdb.set_trace()
        #     print("Exception reading i=%d curr_idx=%d fn=%s" % (i, curr_idx, str(img_path)))
        #     print("Exception was:\n %s" % (str(e)))
        #     if img is not None:
        #         print("img shape %s" % (str(img.shape)))

    for a in annotators:
        a.flush_annotation_set()
    return image_ids_list

if __name__ == "__main__":
    human_face = "/m/0dzct"
    face_images = summarise_class(human_face)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-from",
        help="Continue annotating using annotations from given time stamp",
        required=False, action="store", type=str
    )
    args = parser.parse_args()

    processed = set()
    for segment in ['bbox', 'human_labels', 'machine_labels']:
        if segment in face_images['train']:
            print("Will process", len(face_images['train'][segment]), "training images from", segment)

    for segment in ['bbox', 'human_labels', 'machine_labels']:
        if segment in face_images['train']:
            print("Processing", segment)
            processed.update(bulk_annotate(face_images['train'][segment], processed, continue_from=args.continue_from))
    print("total processed", len(processed))
