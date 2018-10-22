"""
browse and inspect open-images data and added annotations
"""
import sys
import signal 
import random
import numpy as np
import os
import cv2

from PySide2 import QtCore, QtWidgets, QtGui
import qimage2ndarray
from skimage import io

from annotate.data import (
    check_all_files_exist, load_hierarchy, summarise_class, get_img_fn,
    OpenImagesBoxAnnotationSet, OpenImagesBoxAnnotator
)
from annotate.faces import FaceBoundingBoxAnnotator, FaceLandmarkAnnotator
from annotate import settings



class AnnotatorWidget(QtWidgets.QWidget):
    def __init__(self, annotator, *args):
        QtWidgets.QWidget.__init__(self)

        self.annotator = annotator(*args)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setMargin(1)
        self.checkbox = QtWidgets.QCheckBox()
        self.layout.addWidget(self.checkbox, 1)

        self.label = QtWidgets.QLabel()
        self.label.setText(annotator.__name__)
        self.layout.addWidget(self.label, 3)

        self.source = QtWidgets.QComboBox()
        for version in self.annotator.get_versions():
            if version is None:
                self.source.addItem("model")
            else:
                self.source.addItem(version)
        self.layout.addWidget(self.source, 2)

        self.setLayout(self.layout)


class CanvasWidget(QtWidgets.QLabel):
    def __init__(self):
        QtWidgets.QLabel.__init__(self)
        self.setMinimumSize(1,1)

    def resizeEvent(self, size_event):
        w = size_event.size().width()
        h = size_event.size().height()

        #self.canvas.width()
        self.setPixmap(self.img_pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))

    def copyToClipboard(self):
        cb = QtWidgets.QApplication.clipboard()
        cb.setPixmap(self.img_pixmap)

    def contextMenuEvent(self, event):
        Act = QtWidgets.QAction("&Copy", self)
        Act.setShortcuts(QtGui.QKeySequence.New)
        Act.setStatusTip("Create a new file")
        Act.triggered.connect(self.copyToClipboard)

        menu = QtWidgets.QMenu(self)
        menu.addAction(Act)
        menu.exec_(event.globalPos())

class AnnotationWindow(QtWidgets.QWidget):
    def __init__(self, image_ids):
        self.image_ids = image_ids
        self.image_ids_list = list(image_ids.keys())

        QtWidgets.QWidget.__init__(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout_comparison = QtWidgets.QHBoxLayout()
        self.layout_annotators = QtWidgets.QVBoxLayout()

        self.anno_widgets = []
        self.anno_widgets.append(AnnotatorWidget(OpenImagesBoxAnnotator))
        self.anno_widgets.append(AnnotatorWidget(FaceBoundingBoxAnnotator, 'hog'))
        self.anno_widgets.append(AnnotatorWidget(FaceLandmarkAnnotator))

        self.layout_annotators.setSpacing(1)
        for a_widget in self.anno_widgets:
            a_widget.checkbox.clicked.connect(self.refresh_image)
            a_widget.source.currentTextChanged.connect(self.refresh_image)
            self.layout_annotators.addWidget(a_widget)

        self.canvas = CanvasWidget()

        self.layout_comparison.addWidget(self.canvas)

        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left))
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right))

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.addWidget(self.prev_button, 0)
        self.layout_buttons.addWidget(self.next_button, 0)

        self.layout.addLayout(self.layout_annotators, 0)
        self.layout.addLayout(self.layout_comparison, 1)
        self.layout.addLayout(self.layout_buttons, 0)

        self.setLayout(self.layout)

        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)

        self.current_image_idx = 0
        self.img_path = None
        self.next_image()

    def next_image(self):
        img_path = None
        while img_path is None or not os.path.exists(img_path):
            idx = (self.current_image_idx + 1) % len(self.image_ids_list)
            img_id = self.image_ids_list[idx]
            img_path = get_img_fn(img_id)
            self.current_image_idx = idx
        self.img_path = img_path
        self.refresh_image()

    def prev_image(self):
        img_path = None
        while img_path is None or not os.path.exists(img_path):
            idx = (self.current_image_idx - 1) % len(self.image_ids_list)
            img_id = self.image_ids_list[idx]
            img_path = get_img_fn(img_id)
            self.current_image_idx = idx
        self.img_path = img_path
        self.refresh_image()

    def refresh_image(self):
        img_id = self.image_ids_list[self.current_image_idx]
        img = io.imread(self.img_path)
        print(img_id, img.shape)

        if len(img.shape) == 2:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = np.array(img, copy=True)

        all_annotations = {}
        for a in self.anno_widgets:
            if not a.checkbox.isChecked():
                continue

            version_name = a.source.currentText()
            #print(a.annotator.__class__, version_name, all_annotations)
            if version_name == "model":
                annotations = a.annotator.annotate_image(img_id, img, all_annotations)
            else:
                # need to listen for event and load the annotationset then
                # for aset_version in a.annotator.get_versions():
                #     if aset_version is not None and aset_version == version_name:
                #         a.annotator.load_version(aset_version)
                annotations = a.annotator.annotation_set.get_annotation(img_id)
            a.annotator.show_annotation(img_id, rgb_img, annotations)
            #print(annotations)
            all_annotations.update(annotations)
                        
        self.img = qimage2ndarray.array2qimage(rgb_img)
        self.canvas.img_pixmap = QtGui.QPixmap.fromImage(self.img)

        w = self.canvas.width()
        h = self.canvas.height()
        self.canvas.setPixmap(self.canvas.img_pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))


def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\n')
    QtWidgets.QApplication.quit()


if __name__ == "__main__":
    # Inspect classes at http://www.cvdfoundation.org/datasets/open-images-dataset/vis/index.html
    signal.signal(signal.SIGINT, sigint_handler)

    check_all_files_exist()
    class_hierarchy = load_hierarchy()

    human_face = "/m/0dzct"
    face_images = summarise_class(human_face)
    #show_random(face_images['train_bbox'])
    app = QtWidgets.QApplication(sys.argv)

    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    processed = set()
    for segment in ['bbox']: #, 'human_labels', 'machine_labels']:
        print("Will process", len(face_images['train'][segment]), "training images from", segment)

    widget = AnnotationWindow(face_images['train']['bbox'])
    widget.show()

    sys.exit(app.exec_())


