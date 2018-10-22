import sys
import logging
import base64
import io
import os

import skimage.io
from skimage import img_as_uint
import matplotlib.pyplot as plt
import PIL, PIL.Image

from annotate.annotate import Annotator
from annotate.settings import MODEL_DIR, LIB_DIR

log = logging.getLogger(__package__ + "." + __name__)


def fig2png_buffer(fig):
    fig.canvas.draw()

    buffer = io.BytesIO()

    pilImage = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    return buffer


def visualize(img, r, class_names):
    from mrcnn import visualize
    # Visualize results
    fig, ax = plt.subplots(1, figsize=plt.figaspect(img))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax)
    viz_img_buff = fig2png_buffer(fig)

    r["resultImage"] = base64.b64encode(viz_img_buff.getvalue()).decode('ascii')


class MaskRCNN_COCO_Annotator(Annotator):
    # COCO Class names
    # Index of the class in the list is its ID.
    # Need to map these to classes in Google Images
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self):
        # To find local version of the library
        sys.path.append(os.path.join(LIB_DIR, "mask_rcnn"))
        # Import COCO config, local version
        sys.path.append(os.path.join(LIB_DIR, "mask_rcnn/samples/coco/"))

    def load_model(self):
        import tensorflow as tf
        from keras import backend as K

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        K.set_session(sess)

        import mrcnn.model as modellib
        import coco

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        # Directory to save logs and trained model
        OUTPUT_DIR = os.path.join(MODEL_DIR, "logs")
        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=OUTPUT_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        log.info("Mask_RCNN weights loaded and model initialised")
        return sess, model

    def annotate_image(self, img):
        session, model = self.load_model()

        # Run detection
        results = model.detect([img], verbose=1)
        r = results[0]

        r['rois'] = r['rois'].tolist()
        r['class_ids'] = r['class_ids'].tolist()
        r['class_names'] = [self.class_names[i] for i in r['class_ids']]
        r['scores'] = r['scores'].tolist()
        masks = r['masks']
        r['masks'] = []
        for i in range(masks.shape[2]):
            # convert mask arrays into gray-scale pngs, then base64 encode them
            buff = io.BytesIO()
            skimage.io.imsave(buff, img_as_uint(masks[:, :, i]))
            b64img = base64.b64encode(buff.getvalue()).decode('ascii')
            r['masks'].append(b64img)

        # This is perhaps an in vain attempt to free GPU memory
        del model
        session.close()
        return r