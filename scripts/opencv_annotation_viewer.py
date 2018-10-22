"""
browse and inspect open-images data and added annotations using opencv

This is deprecated in favour of the Qt based UI.

Qt UI clashes with OpenCV GUI dependencies, so to use this you need to
install the opencv-contrib-python package from pip instead of opencv-contrib-python-headless
(which is in requirements.txt)

"""
from skimage import io
import numpy as np
import os
import random
import cv2
import dlib

from annotate.data import check_all_files_exist, load_hierarchy, summarise_class
from annotate import FaceBoundingBoxAnnotator, FaceLandmarkAnnotator
from annotate import settings


def show_random(image_ids):
    image_ids_list = list(image_ids.keys())

    face_bbox = FaceBoundingBoxAnnotator()
    face_landmark = FaceLandmarkAnnotator()

    cv2.namedWindow("Preview")
    loop = True
    while loop:
        idx = random.randint(0, len(image_ids_list) - 1)
        img_id = image_ids_list[idx]
        img_path = os.path.join(settings.OPEN_IMAGES_DIR, 'train', img_id + '.jpg')
        print(img_id)
        if os.path.exists(img_path):
            img = io.imread(img_path)
            print(img.shape)

            dets = face_bbox.annotate_image(img_id, img, {})
            landmarks = face_landmark.annotate_image(img_id, img, {"face_bbox": dets}) 

            if len(img.shape) == 2:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = img

            face_bbox.show_annotation(img_id, rgb_img, dets)
            face_landmark.show_annotation(img_id, rgb_img, landmarks)

            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Preview", bgr_img)
            k = -1
            while k == -1:
                try:
                    k = cv2.waitKeyEx(100)
                    print(k)
                    if k == ord('q') or cv2.getWindowProperty("Preview",cv2.WND_PROP_VISIBLE) < 1:        
                        loop = False
                        break
                except:
                    loop = False
                    break
        else:
            print("No such file", img_path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Inspect classes at http://www.cvdfoundation.org/datasets/open-images-dataset/vis/index.html

    check_all_files_exist()
    class_hierarchy = load_hierarchy()

    human_face = "/m/0dzct"
    face_images = summarise_class(human_face)

    show_random(face_images['train_bbox'])
