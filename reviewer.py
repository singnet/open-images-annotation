"""
review annotations and human rate them
"""
import sys
import math
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



class AnnotationSelectWidget(QtWidgets.QWidget):
    def __init__(self, annotators):
        QtWidgets.QWidget.__init__(self)

        self.annotators = annotators

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setMargin(1)
        self.annotator_combobox = QtWidgets.QComboBox()
        for annotator in self.annotators:
            self.annotator_combobox.addItem(annotator.__class__.__name__, annotator)

        self.layout.addWidget(self.annotator_combobox, 1)

        self.annotator_combobox.currentTextChanged.connect(self.load_versions)
        self.annotator = None
        self.human_annotation_set = None

        self.source = QtWidgets.QComboBox()
        self.layout.addWidget(self.source, 2)

        self.source.currentTextChanged.connect(self.load_version)
        self.setLayout(self.layout)
        self.load_versions(None)

    def load_versions(self, event):
        #for a in self.annotators:
        a = self.annotator_combobox.currentData()
        self.source.clear()
        for version in a.get_versions():
            if version is not None:
                self.source.addItem(version)
        a = self.annotator_combobox.currentData()
    
    def load_version(self, event):
        print(event)
        a = self.annotator_combobox.currentData()
        fn = self.source.currentText()
        a.load_version(fn)
        self.annotator = a
        self.human_annotation_set = a.get_human_annotations()



class CanvasWidget(QtWidgets.QLabel):
    def __init__(self, parent, zoomed=False):
        QtWidgets.QLabel.__init__(self)
        self.setMinimumSize(1,1)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.zoomed = zoomed 

        self.img_pixmap = None

        self.p = parent
        self.t = None
        self.points = []

        self.selected_idx = None

    def resizeEvent(self, size_event):
        self.refresh()

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

    def mousePressEvent(self, QMouseEvent):
        if self.t is None:
            return
        mx = QMouseEvent.x()
        my = QMouseEvent.y()
        closest_idx = None
        closest_distance = math.inf
        closest_point = None
        for idx, pos in enumerate(zip(self.points[::2], self.points[1::2])):
            distsqr = (pos[0] - mx)**2 + (pos[1] - my)**2
            if distsqr < closest_distance:
                closest_idx = idx
                closest_distance = distsqr
                closest_point = pos
        #print("closest point is idx %d, %s - distance %.4f" % (closest_idx, str(closest_point), math.sqrt(float(closest_distance))))
        self.selected_idx = closest_idx
        if not self.p.human_button.isChecked():
            if self.p.human_annotations is None: 
                self.p.human_annotations = {'face_landmarks': []}
            # TODO need to check there isn't already a matching human landmark
            source_anno = self.p.annotations['face_landmarks'][self.p.current_annotation_idx]
            self.p.human_annotations['face_landmarks'].append(source_anno)
            self.p.human_button.setChecked(True)
            self.p.current_annotation_idx = len(self.p.human_annotations['face_landmarks']) - 1

    def update_selected_point(self, pt):
        new_point = (
            (pt[0]/self.t[1]) - self.t[0][0],
            (pt[1]/self.t[1]) - self.t[0][1],
        )

        w = self.img.shape[1]
        h = self.img.shape[0]

        self.p.human_annotations['face_landmarks'][self.p.current_annotation_idx][3][self.selected_idx * 2] = new_point[0] / w
        self.p.human_annotations['face_landmarks'][self.p.current_annotation_idx][3][self.selected_idx * 2 + 1] = new_point[1] / h

    def mouseMoveEvent(self, QMouseEvent):
        if self.selected_idx is None:
            return
        pt = (QMouseEvent.x(), QMouseEvent.y())
        self.update_selected_point(pt)
        self.refresh()

    def mouseReleaseEvent(self, QMouseEvent):
        self.p.human_annotation_set.set_annotation(self.p.img_id, self.p.human_annotations['face_landmarks'])
        self.selected_idx = None

    def refresh(self):
        w = self.width()
        h = self.height()

        if self.img is None:
            return

        if self.p.annotator is not None:
            a = self.p.annotator
            if self.p.human_button.isChecked():
                annotations = self.p.human_annotations
                colour = (50, 190, 255)
            else:
                annotations = self.p.annotations
                colour = self.p.annotator.colour

            if annotations is not None:
                if self.zoomed:
                    # create zoom image
                    self.t, self.points, img = a.show_annotation(self.p.img_id, self.img, annotations, self.p.current_annotation_idx, zoom=True, target_size=(w, h), colour=colour)

                    qimg = qimage2ndarray.array2qimage(img)
                    self.img_pixmap = QtGui.QPixmap.fromImage(qimg)
                    self.setPixmap(self.img_pixmap)
                else:
                    # show on full size image
                    _t, _points, img = a.show_annotation(self.p.img_id, self.img, annotations, self.p.current_annotation_idx, colour=colour)
                        
                    qimg = qimage2ndarray.array2qimage(img)
                    self.img_pixmap = QtGui.QPixmap.fromImage(qimg)
                    self.setPixmap(self.img_pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))
            else:
                print("No annotations")
                qimg = qimage2ndarray.array2qimage(self.img)
                self.img_pixmap = QtGui.QPixmap.fromImage(qimg)
                self.setPixmap(self.img_pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio))


class AnnotationWindow(QtWidgets.QWidget):
    def __init__(self, image_ids):
        self.image_ids = image_ids
        self.image_ids_list = list(image_ids.keys())
        self.annotator = None
        self.annotations = None
        self.human_annotations = None
        self.current_image_idx = 0
        self.current_annotation_idx = 0
        self.img_id = None
        self.img_path = None
        self.rgb_img = None

        QtWidgets.QWidget.__init__(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout_comparison = QtWidgets.QHBoxLayout()
        self.layout_annotators = QtWidgets.QVBoxLayout()

        self.annotators = [FaceLandmarkAnnotator(), FaceBoundingBoxAnnotator('hog')]
        #self.anno_widgets.append(AnnotationSelectWidget([OpenImagesBoxAnnotator))
        self.annotation_select = AnnotationSelectWidget(self.annotators)

        self.layout_annotators.setSpacing(1)
        self.annotation_select.source.currentTextChanged.connect(self.change_annotation_set)
        self.layout_annotators.addWidget(self.annotation_select)

        self.canvas = CanvasWidget(self)
        self.canvas_zoom = CanvasWidget(self, zoomed=True)

        self.layout_comparison.addWidget(self.canvas)
        self.layout_comparison.addWidget(self.canvas_zoom)

        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left))
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right))

        self.prev_a_button = QtWidgets.QPushButton("Previous Anno")
        self.prev_a_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_A))
        self.next_a_button = QtWidgets.QPushButton("Next Anno")
        self.next_a_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_B))

        self.human_button = QtWidgets.QPushButton("Show Human")
        self.human_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H))
        self.human_button.setCheckable(True)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_S))

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.addWidget(self.prev_button, 0)
        self.layout_buttons.addWidget(self.next_button, 0)
        self.layout_buttons.addWidget(self.prev_a_button, 0)
        self.layout_buttons.addWidget(self.next_a_button, 0)
        self.layout_buttons.addWidget(self.human_button, 0)
        self.layout_buttons.addWidget(self.save_button, 0)

        self.layout.addLayout(self.layout_annotators, 0)
        self.layout.addLayout(self.layout_comparison, 1)
        self.layout.addLayout(self.layout_buttons, 0)

        self.setLayout(self.layout)

        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_a_button.clicked.connect(self.next_anno)
        self.prev_a_button.clicked.connect(self.prev_anno)
        self.human_button.clicked.connect(self.switch_human)
        self.save_button.clicked.connect(self.save_human)

        self.change_annotation_set(None)

    def switch_human(self, event):
        self.change_anno(0)
        self.refresh_image()

    def save_human(self, event):
        print("saving")
        self.human_annotation_set.flush()

    def change_image(self, offset):
        img_path = None
        while img_path is None or not os.path.exists(img_path):
            idx = (self.current_image_idx + offset) % len(self.image_ids_list)
            img_id = self.image_ids_list[idx]
            img_path = get_img_fn(img_id)
            self.current_image_idx = idx
        self.current_annotation_idx = 0
        self.img_path = img_path

        if self.human_button.isChecked():
            self.human_button.setChecked(False)

        self.img_id = self.image_ids_list[self.current_image_idx]
        img = io.imread(self.img_path)

        if self.annotator is not None:
            self.annotations = self.annotator.annotation_set.get_annotation(self.img_id)
            self.human_annotations = self.human_annotation_set.get_annotation(self.img_id)

        if len(img.shape) == 2:
            self.rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            self.rgb_img = np.array(img, copy=True)
        self.canvas.img = np.array(self.rgb_img, copy=True)
        self.canvas_zoom.img = np.array(self.rgb_img, copy=True)
        

    def change_annotation_set(self, event):
        self.annotator = self.annotation_select.annotator
        self.human_annotation_set = self.annotation_select.human_annotation_set

        self.current_annotation_idx=0
        self.current_image_idx=0
        self.change_image(0)
        self.refresh_image()

    def next_image(self):
        self.change_image(1)
        self.refresh_image()

    def prev_image(self):
        self.change_image(-1)
        self.refresh_image()

    def change_anno(self, offset):
        if self.annotator is not None:
            if self.human_button.isChecked():
                annotations = self.human_annotations
            else:
                annotations = self.annotations

            self.current_annotation_idx += offset
            if self.current_annotation_idx >= self.annotator.count_annotations(annotations):
                if offset > 0:
                    self.change_image(1)
                else:
                    self.current_annotation_idx = self.annotator.count_annotations(annotations) - 1
            elif self.current_annotation_idx < 0:
                self.change_image(-1)
                # change image resets to machine labels
                annotations = self.annotations
                # We want to show the last annotation on the previous image otherwise
                # going forward then back won't result in us being at the same place!
                self.current_annotation_idx = self.annotator.count_annotations(annotations) - 1
        else:
            self.current_annotation_idx = 0
        if annotations is None:
            print("img id:", self.img_id, self.rgb_img.shape, ", human:", self.human_button.isChecked(), ", current anno idx:", self.current_annotation_idx, ", No annotations")
        else:
            print("img id:", self.img_id, self.rgb_img.shape, ", human:", self.human_button.isChecked(), ", current anno idx:", self.current_annotation_idx, ", num annotations", len(annotations.get('face_landmarks', [])))


    def next_anno(self):
        self.change_anno(1)
        self.refresh_image()

    def prev_anno(self):
        self.change_anno(-1)
        self.refresh_image()

    def refresh_image(self):
        self.canvas.refresh()
        self.canvas_zoom.refresh()


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


