from os.path import getsize, basename
from tqdm import tqdm

import logging
import csv
import os
import json
import random
import cv2
import numpy as np

from memorize import Memorize
Memorize.USE_CURRENT_DIR = False

import dlib
import numpy as np
from annotate import settings
from annotate.annotate import AnnotationSet, Annotator
from annotate.utils import PositionBasedCSVReader, build_csv_index
from annotate._data import count_class

# TODO find better pattern for testing
DEBUG = False
#DEBUG = True

human_labels = [
    "/m/014sv8,Human eye",
    "/m/015h_t,Human beard",
    "/m/0283dt1,Human mouth",
    "/m/02p0tk3,Human body",
    "/m/031n1,Human foot",
    "/m/035r7c,Human leg",
    "/m/039xj_,Human ear",
    "/m/03q69,Human hair",
    "/m/04hgtk,Human head",
    "/m/0dzct,Human face",
    "/m/0dzf4,Human arm",
    "/m/0k0pj,Human nose",
    "/m/0k65p,Human hand",
]


class OpenImagesBoxAnnotator(Annotator):

    def __init__(self):
        super().__init__()

        self.provides = ["bbox"]
        self.annotation_set = OpenImagesBoxAnnotationSet()
        self.colour = (200, 200, 100)

    def load_version(self, version):
        pass

    def get_versions(self):
        return [""]

    def show_annotation(self, img_id, img, annotations, index=None):
        from annotate.faces import render_bounding_boxes
        human_face = "/m/0dzct"
        faces = [f[1] for f in annotations['bbox'] if f[0] == human_face]
        render_bounding_boxes(img, faces, self.colour)



class OpenImagesBoxAnnotationSet(AnnotationSet):
    bounding_box_data_files = {
        "bboxes": {
            "train": "train-annotations-bbox.csv",
            "valid": "validation-annotations-bbox.csv",
            "test": "test-annotations-bbox.csv",
        },
        "human_labels": {
            "train": "train-annotations-human-imagelabels-boxable.csv",
            "valid": "validation-annotations-human-imagelabels-boxable.csv",
            "test": "test-annotations-human-imagelabels-boxable.csv",
        },
        "image_ids": {
            "train": "train-images-boxable-with-rotation.csv",
            "valid": "validation-images-with-rotation.csv",
            "test": "test-images-with-rotation.csv",
        },
        "class_descriptions": "class-descriptions-boxable.csv",
        "class_count": 600,
        "class_hierarchy": "bbox_labels_600_hierarchy.json",
        "image_counts": {
            "train": 1743042,
            "valid": 41620,
            "test": 125436,
        },
        "box_counts": {
            "train": 14610229,
            "valid": 204621,
            "test": 625282,
        }
    }

    def __init__(self):
        self.bbox_index, self.last_id = build_csv_index(self._get_file_name())

    def get_version_name(self):
        return "OpenImages v4"

    def _get_file_name(self, segment="train"):
        if DEBUG:
            return get_metadata_file("train-annotations-bbox-debug.csv")
        return get_metadata_file(self.bounding_box_data_files["bboxes"][segment])

    def get_annotation(self, img_id):
        boxes = []
        position = self.bbox_index[img_id]
        with open(self._get_file_name(), 'r') as csvfile:
            reader = PositionBasedCSVReader(csvfile, delimiter=',', quotechar='"')
            reader.seek(position)
            try:
                position, row = next(reader)
                while position and row[0] == img_id:
                    boxes.append(
                        (
                            row[2], # class id
                            (
                                float(row[4]), # left
                                float(row[6]), # top
                                float(row[5]), # right
                                float(row[7]), # bottom
                            )
                        )
                    )
                    position, row = next(reader)
            except StopIteration:
                pass
        #return boxes
        return {'bbox' : boxes}
    
    def rate_annotation(self, img_id, rating, version=None, index=None):
        pass

    def edit_annotation(self, img_id, img, annotations, index, window_context):
        pass

    def flush(self):
        pass


human_and_mammals_data_files = {
    "image_ids": "train-image-ids-with-human-parts-and-mammal-boxes.txt",
    "class_descriptsion": "class-ids-human-body-parts-and-mammal.txt",
}

human_labels_only_data_files = {
    "human_labels": {
        "train": "train-annotations-human-imagelabels.csv",
        "valid": "validation-annotations-human-imagelabels.csv",
        "test": "test-annotations-human-imagelabels.csv",
    },
    "image_ids": {
        "train": "train-images-with-labels-with-rotation.csv",
        "valid": "validation-images-with-rotation.csv",
        "test": "test-images-with-rotation.csv",
    },
    "class_descriptions": "class-descriptions.csv",
    "trainable_classes": "classes-trainable.txt",
    "class_count": 19794,
    "trainable_class_count": 7186,
    "image_counts": {
        "train": 5655108,
        "valid": 41620,
        "test": 125436,
    }
}

machine_labels_only_data_files = {
    "machine_labels": {
        "train": "train-annotations-machine-imagelabels.csv",
        "valid": "validation-annotations-machine-imagelabels.csv",
        "test": "test-annotations-machine-imagelabels.csv",
    },
    "image_ids": {
        "train": "train-images-with-labels-with-rotation.csv",
        "valid": "validation-images-with-rotation.csv",
        "test": "test-images-with-rotation.csv",
    },
    "class_descriptions": "class-descriptions.csv",
    "trainable_classes": "classes-trainable.txt",
    "class_count": 7870,
    "trainable_class_count": 4764,
    "image_counts": {
        "train": 8853429,
        "valid": 41620,
        "test": 125436,
    }
}

all_images_files = {
    "image_ids": "image_ids_and_rotation.csv",
    "image_counts": 9178275
}


def check_all_files_exist():
    print("Checking all OpenImage annotation files exist")
    missing = []
    missing.extend(check_files_exist(human_and_mammals_data_files))
    missing.extend(check_files_exist(OpenImagesBoxAnnotationSet.bounding_box_data_files))
    missing.extend(check_files_exist(human_labels_only_data_files))
    missing.extend(check_files_exist(machine_labels_only_data_files))
    missing.extend(check_files_exist(all_images_files))
    if len(missing) > 0:
        print("Missing files:", missing)
        raise(Exception("annotation files are missing"))


def check_files_exist(data_segment):
    missing = []
    for k, v in data_segment.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(v, str) and (vv.endswith('.csv') or vv.endswith('.txt')):
                    if not os.path.exists(get_metadata_file(vv)):
                        missing.append(vv)
        elif isinstance(v, str) and (v.endswith('.csv') or v.endswith('.txt')):
            if not os.path.exists(get_metadata_file(v)):
                missing.append(v)
    return missing


def get_img_fn(img_id):
    return os.path.join(
        settings.OPEN_IMAGES_DIR_MAP[img_id[0]], 'train', 'train_' + img_id[0],
        img_id + '.jpg'
    )

def get_metadata_file(fn):
    return os.path.join(
        settings.OPEN_IMAGES_DIR_MAP['meta'], fn
    )

def load_hierarchy():
    with open(get_metadata_file(OpenImagesBoxAnnotationSet.bounding_box_data_files['class_hierarchy']), 'rb') as f:
        h = json.load(f)
    return h


def summarise_class(class_label):
    all_images = {
        "train": {},
        "valid": {},
        "test": {},
    }
    print("Summarizing for", class_label)

    # load annotations.
    # there are multiple sources:
    # - bboxes
    # - human labels
    # - machine label
    # 
    # For now we will just work on the smaller bounding box dataset

    for segment in ["train", "valid", "test"]:
        fn = get_metadata_file(OpenImagesBoxAnnotationSet.bounding_box_data_files["bboxes"][segment])
        if DEBUG and segment == "train":
            fn = get_metadata_file("train-annotations-bbox-debug.csv")

        images_seen, count, multiples = count_class(fn, class_label)
        print("bbox", segment, count, multiples)
        all_images[segment]["bbox"] = images_seen
        
    # for segment in ["train", "valid", "test"]:
    #     fn = os.path.join(settings.OPEN_IMAGES_DIR, human_labels_only_data_files["human_labels"][segment])
    #     images_seen, count, multiples = count_class(fn, class_label)
    #     print("human labels", segment, count, multiples)
    #     all_images[segment]["human_labels"] = images_seen
    # for segment in ["train", "valid", "test"]:
    #     fn = os.path.join(settings.OPEN_IMAGES_DIR, machine_labels_only_data_files["machine_labels"][segment])
    #     images_seen, count, multiples = count_class(fn, class_label)
    #     print("machine labels", segment, count, multiples)
    #     all_images[segment]["machine_labels"] = images_seen

    return all_images