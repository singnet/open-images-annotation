import numpy as np
import datetime
import os
import math
import cv2
import dlib
import csv
import itertools

from annotate import settings
from annotate.utils import build_csv_index, PositionBasedCSVReader
from annotate.annotate import Annotator, AnnotationSet, AnnotationTrainer


def render_landmarks_direct(frame, detected_landmarks):
    for l in detected_landmarks:
        landmarks = np.matrix([[p.x, p.y] for p in l])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # annotate the positions
            cv2.putText(frame, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))

            # draw points on the landmark positions
            cv2.circle(frame, pos, 3, color=(0, 255, 255))


def render_landmarks(frame, detected_landmarks, colour):
    w = frame.shape[1]
    h = frame.shape[0]
    for l in detected_landmarks:
        for idx, point in enumerate(zip(l[::2],l[1::2])):
            pos = (int(point[0] * w), int(point[1] * h))

            # annotate the positions
            # cv2.putText(frame, str(idx), pos,
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.4,
            #             color=colour)

            # draw points on the landmark positions
            cv2.circle(frame, pos, 3, color=colour)


def render_bounding_boxes(frame, detected_boxes, colour):
    w = frame.shape[1]
    h = frame.shape[0]
    for i, d in enumerate(detected_boxes):
        cv2.rectangle(frame, (int(d[0] * w), int(d[1] * h)),

                      (int(d[2] * w), int(d[3] * h)), colour, 2)
        # cv2.rectangle(frame, (d.left(), d.top()),
        #               (d.right(), d.bottom()), (0, 255, 0), 2)


class FaceLandmarkAnnotationSet(AnnotationSet):
    @classmethod
    def get_versions(cls):
        import glob
        expression = os.path.join(settings.ANNOTATION_DIR, "face_landmarks_*.csv")
        return [i for i in glob.glob(expression)]

    def __init__(self, file_name=None):
        super().__init__()

        self.backing_store_file = file_name
        if self.backing_store_file is None:
            self.backing_store_file = os.path.join(settings.ANNOTATION_DIR, "face_landmarks_{}.csv".format(datetime.datetime.now().isoformat().split('.')[0]))
        self._index = None
        if os.path.exists(self.backing_store_file):
            self._index, self.last_id = build_csv_index(self.backing_store_file)

        self._cache = {}

    def get_annotation(self, img_id, version=None, index=None):
        if img_id in self._cache:
            return {'face_landmarks': self._cache[img_id]}

        if not os.path.exists(self.backing_store_file):
            return None
            
        if self._index is None:
            self._index, self.last_id = build_csv_index(self.backing_store_file)

        if img_id not in self._index:
            return None
        annotations = []
        position = self._index[img_id]
        with open(self.backing_store_file, 'r') as csvfile:
            reader = PositionBasedCSVReader(csvfile, delimiter=',', quotechar='"')
            reader.seek(position)
            try:
                position, row = next(reader)
                while row[0] == img_id:
                    bbox = [float(x) for x in row[2:6]]
                    if max(bbox) == 0.0:
                        bbox = None
                    face_bbox = [float(x) for x in row[6:10]]
                    if max(face_bbox) == 0.0:
                        face_bbox = None
                    annotations.append((
                        row[1],
                        bbox,
                        face_bbox,
                        [float(x) for x in row[10:]]
                    ))
                    position, row = next(reader)
            except StopIteration:
                pass
        return {'face_landmarks': annotations}

    def set_annotation(self, img_id, annotations):
        self._cache[img_id] = annotations

    def flush(self):
        fn = self.backing_store_file
        for img_id in sorted(self._cache.keys()):
            if not os.path.exists(fn):
                with open(fn,'w') as f:
                    header = [
                        'img_id','source',
                        'bbox_left','bbox_top','bbox_right','bbox_bottom',
                        'face_bbox_left','face_bbox_top','face_bbox_right','face_bbox_bottom',
                        ]
                    lm_idx = range(1,69)
                    header.extend(list(itertools.chain.from_iterable([(str(i) + 'x', str(i) + 'y') for i in lm_idx])))
                    header_str = ','.join(header) + "\n"
                    f.write(header_str)
            with open(fn,'a',newline='') as f:
                writer=csv.writer(f)
                for source, bbox, face_bbox, lm in self._cache[img_id]:
                    if bbox is None:
                        bbox = [0.0, 0.0, 0.0, 0.0]
                    if face_bbox is None:
                        face_bbox = [0.0, 0.0, 0.0, 0.0]
                    p = (
                        [source] +
                        [("%.5f" % b) for b in bbox] +
                        [("%.5f" % b) for b in face_bbox] +
                        [("%.5f" % l) for l in lm]
                    )
                    p.insert(0, img_id)
                    writer.writerow(p)
        self._index = None
        self._cache = {}


class FaceLandmarkAnnotator(Annotator):

    def __init__(self):
        super().__init__()
        # how to train shape predictors: http://dlib.net/train_shape_predictor_ex.cpp.html

        # landmark prediction
        landmark68_predictor_path = os.path.join(settings.MODEL_DIR, "shape_predictor_68_face_landmarks_oi_hog_bbox_5000.dat")
        self.landmark68_predictor = dlib.shape_predictor(landmark68_predictor_path)

        self.landmark_predictors = [self.landmark68_predictor]
        self.landmark_idx = 0

        self.border = 0.2

        self.requires = ["face_bbox"]
        self.provides = ["face_landmarks"]
        self.annotation_set = FaceLandmarkAnnotationSet()
        self.colour = (0, 255, 80)

        self.segments = [
            list(range(17)),
            list(range(17, 22)),
            list(range(22, 27)),
            list(range(27, 31)),
            list(range(31, 36)),
            list(range(36, 42)) + [36],
            list(range(42, 48)) + [42],
            list(range(48, 60)) + [48],
            list(range(60, 65)),
            list(range(64, 68)) + [60]
        ]

    def get_human_annotations(self):
        human_annotations = os.path.join(settings.ANNOTATION_DIR, "face_landmarks_human.csv")

        return FaceLandmarkAnnotationSet(human_annotations)

    def load_version(self, version):
        self.annotation_set = FaceLandmarkAnnotationSet(version)

    def get_versions(self):
        return [None] + FaceLandmarkAnnotationSet.get_versions()

    def merge_dets(self, face_dets, oi_dets):
        # Returns a single list of paired dets.
        # If a overlap matches, return (face_det, oi_det)
        # If not, return (None, oi_det) or (face_det, None)
        def centroid(det):
            centre_x = face_det[2] - face_det[0]
            centre_y = face_det[3] - face_det[1]
            return centre_x, centre_y

        all_dets = []
        matches = 0
        misses = 0

        for (_, face_det) in face_dets:
            best_oi_det = None
            centre_x, centre_y = centroid(face_det)
            face_area = math.sqrt((face_det[2] - face_det[0]) * (face_det[3] - face_det[1]))
            #print("face_det", face_det, face_area)
            for i, (_, oi_det) in enumerate(oi_dets):
                # get overlap
                oi_area = math.sqrt((oi_det[2] - oi_det[0]) * (oi_det[3] - oi_det[1]))
                x_overlap = max(0, min(face_det[2], oi_det[2]) - max(face_det[0], oi_det[0]))
                y_overlap = max(0, min(face_det[3], oi_det[3]) - max(face_det[1], oi_det[1]))
                overlap_area = math.sqrt(x_overlap * y_overlap)
                #print("   oi_det", oi_det, oi_area)
                #print("    overlap", overlap_area)
                if overlap_area > (0.4*oi_area) and overlap_area > (0.4*face_area):
                    # get centroid distance
                    oi_centre_x, oi_centre_y = centroid(oi_det)
                    dist = ((oi_centre_x - centre_x) * (oi_centre_x - centre_x)) + ((oi_centre_y - centre_y) * (oi_centre_y - centre_y))
                    if best_oi_det is None:
                        #print("            match")
                        best_oi_det = (i, dist, overlap_area, oi_det)
                    elif dist < best_oi_det[0] and overlap_area > best_oi_det[1]:
                        #print("            match")
                        best_oi_det = (i, dist, overlap_area, oi_det)

            if best_oi_det is not None:
                idx, _, _, det = best_oi_det
                all_dets.append((face_det, det))
                del oi_dets[idx]
                matches += 1
            else:
                all_dets.append((face_det, None))
                misses += 1

        for _source, d in oi_dets:
            all_dets.append((None, d))
            misses += 1
        #print("matches", matches)
        #print("misses", misses)

        return all_dets
    
    def _find_face_landmarks(self, img, prior_inputs):
        # somehow resolve human bbox with predicted?
        # if bbox is more than 70% overlapping, take the larger of the two
        # if bbox is less than 70%, keep both?
        # possibly get face bbox annotator to do the merging and then save those?
        dets = []
        face_dets = []
        oi_dets = []
        if len(prior_inputs.get('face_bbox', [])) > 0:
            face_dets = prior_inputs["face_bbox"]
            face_dets = [('face_bbox', d) for d in face_dets]
        if 'bbox' in prior_inputs:
            human_face = "/m/0dzct"
            oi_dets = [('bbox', f[1]) for f in prior_inputs["bbox"] if f[0] == human_face]
        dets = self.merge_dets(face_dets, oi_dets)

        landmarks = [None] * len(dets)

        if img.shape[-1] == 4:
            img = img[:, :, :3]
            
        for i, (face_bbox, bbox) in enumerate(dets):
            if face_bbox is not None:
                source = 'face_bbox'
                d = face_bbox
            else:
                source = 'bbox'
                d = bbox

            landmark_predictor = self.landmark_predictors[self.landmark_idx]
            det = dlib.rectangle(
                int(d[0] * img.shape[1]),
                int(d[1] * img.shape[0]),
                int(d[2] * img.shape[1]),
                int(d[3] * img.shape[0]),
            )
            landmarks_object = landmark_predictor(img, det)

            l = landmarks_object.parts()

            # convert positions to 0..1
            px = [float(p.x) / img.shape[1] for p in l]
            py = [float(p.y) / img.shape[0] for p in l]
            p = list(itertools.chain.from_iterable(zip(px, py)))

            landmarks[i] = (source, bbox, face_bbox, p)

        return landmarks

    def annotate_image(self, img_id, img, prior_inputs):
        x = self._find_face_landmarks(img, prior_inputs)
        self.annotation_set.set_annotation(img_id, x)
        return {'face_landmarks': x}

    def count_annotations(self, annotations):
        if annotations is None:
            return 0
        return len(annotations['face_landmarks'])

    def show_single_annotation(self, img_id, img, anno, colour):
        source = anno[0]
        if source == 'face_bbox':
            render_bounding_boxes(img, [anno[2]], colour)
        else:
            render_bounding_boxes(img, [anno[1]], colour)
        render_landmarks(img, [anno[3]], colour)

    def show_annotation(self, img_id, img, annotations, index=None, zoom=False, target_size=None, colour=None):
        if colour is None:
            colour = self.colour

        points = None
        transform = None
        if index is not None:
            if zoom:
                from skimage.transform import rescale
                anno = annotations['face_landmarks'][index]
                source = anno[0]
                if source == 'face_bbox':
                    d = anno[2]
                else:
                    d = anno[1]

                w = img.shape[1]
                h = img.shape[0]
                d = (int(d[0] * w), int(d[1] * h), int(d[2] * w), int(d[3] * h))
                b = int((d[2] - d[0]) * 0.25)
                crop_box = (max(0, d[0] - b), max(0, d[1] - b), min(w, d[2] + b), min(h, d[3] + b))
                img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
                d2 = (d[0] - min(d[0], crop_box[0]), d[1] - min(d[1], crop_box[1]), d[2] - min(d[0], crop_box[0]), d[3] - min(d[1], crop_box[1]))
                rescale_factor = min(target_size[0] / float(img.shape[1]), target_size[1] / float(img.shape[0]))
                img = (255 * rescale(img, rescale_factor, anti_aliasing=True)).astype(np.uint8)

                cv2.rectangle(img,
                    ( int(d2[0]*rescale_factor), int(d2[1]*rescale_factor) ),
                    ( int(d2[2]*rescale_factor), int(d2[3]*rescale_factor) ),
                    colour, 2
                    )
                l = list(anno[3])
                l_t = list(l)

                for idx in range(len(l) // 2):
                    point = (l[2*idx], l[2*idx+1])
                    pos = (
                        int((int(point[0] * w) - min(d[0], crop_box[0])) * rescale_factor),
                        int((int(point[1] * h) - min(d[1], crop_box[1])) * rescale_factor)
                        )
                    l_t[2*idx] = pos[0]
                    l_t[2*idx+1] = pos[1]

                for s in self.segments:
                    last_pos = None
                    for idx in s:
                        pos = (l_t[2*idx], l_t[2*idx+1])
                        if last_pos is not None:
                            cv2.line(img, last_pos, pos, colour, 1)
                        last_pos = pos

                for idx, pos in enumerate(zip(l_t[::2],l_t[1::2])):
                    cv2.circle(img, pos, 4, color=colour)

                points = l_t
                transform = ((-min(d[0], crop_box[0]), -min(d[1], crop_box[1])), rescale_factor)
            else:
                img = np.array(img, copy=True)
                self.show_single_annotation(img_id, img, annotations['face_landmarks'][index], colour=colour)
        else:
            img = np.array(img, copy=True)
            for anno in annotations['face_landmarks']:
                self.show_single_annotation(img_id, img, anno, colour=colour)
        return transform, points, img


class FaceBoundingBoxAnnotationSet(AnnotationSet):
    @classmethod
    def get_versions(cls):
        import glob
        expression = os.path.join(settings.ANNOTATION_DIR, "face_bbox_*.csv")
        return [i for i in glob.glob(expression)]

    def __init__(self, file_name=None, prefix='face_bbox'):
        super().__init__()

        self.backing_store_file = file_name
        if self.backing_store_file is None:
            self.backing_store_file = os.path.join(settings.ANNOTATION_DIR, "{}_{}.csv".format(prefix,datetime.datetime.now().isoformat().split('.')[0]))
        self._index = None
        if os.path.exists(self.backing_store_file):
            self._index, self.last_id = build_csv_index(self.backing_store_file)

        self._cache = {}

    def get_annotation(self, img_id, version=None, index=None):
        if img_id in self._cache:
            return {'face_bbox': self._cache[img_id]}

        if not os.path.exists(self.backing_store_file):
            return None
            
        if self._index is None:
            self._index, self.last_id = build_csv_index(self.backing_store_file)

        if img_id not in self._index:
            return None
        annotations = []
        position = self._index[img_id]
        with open(self.backing_store_file, 'r') as csvfile:
            reader = PositionBasedCSVReader(csvfile, delimiter=',', quotechar='"')
            reader.seek(position)
            try:
                position, row = next(reader)
                while row[0] == img_id:
                    annotations.append(
                        [float(x) for x in row[1:]]
                    )
                    position, row = next(reader)
            except StopIteration:
                pass
        return {'face_bbox': annotations}

    def set_annotation(self, img_id, annotations):
        self._cache[img_id] = annotations

    def flush(self):
        fn = self.backing_store_file
        for img_id in sorted(self._cache.keys()):
            if not os.path.exists(fn):
                with open(fn,'w') as f:
                    f.write('img_id,left,top,right,bottom\n')
            with open(fn,'a',newline='') as f:
                writer=csv.writer(f)
                for b in self._cache[img_id]:
                    writer.writerow([img_id, b[0], b[1], b[2], b[3]])
        self._index = None
        self._cache = {}


class FaceBoundingBoxAnnotator(Annotator):

    def __init__(self, method='cnn'):
        super().__init__()

        if method == 'cnn':
            print("Using CNN face detector")
            # training described at top of http://dlib.net/dnn_mmod_face_detection_ex.cpp.html
            # and at http://blog.dlib.net/2016/10/easily-create-high-quality-object.html
            cnn_face_detector_path = os.path.join(settings.MODEL_DIR, "dlib_cnn_mmod_human_face_detector_v1.dat")
            self.face_detector_dlib_cnn = dlib.cnn_face_detection_model_v1(
                cnn_face_detector_path)
            self.prefix = 'face_bbox'
            self.face_idx = 2
        elif method == 'hog':
            print("Using HOG face detector")
            # face detection/localization
            # training: http://dlib.net/train_object_detector.py.html
            self.face_detector_dlib_hog = dlib.get_frontal_face_detector()
            self.prefix = 'face_bbox_hog'
            self.face_idx = 1
        elif method == 'haar':
            print("Using Haar face detector")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector_opencv_haarcascade = cv2.CascadeClassifier(cascade_path)
            self.prefix = 'face_bbox_haar'
            self.face_idx = 0
        else:
            raise Exception("Invalid method %s" % (str(method),))


        self.border = 0.2

        self.provides = ['face_bbox']
        self.annotation_set = FaceBoundingBoxAnnotationSet(prefix=self.prefix)
        self.colour = (20, 255, 80)

    def load_version(self, version):
        self.annotation_set = FaceBoundingBoxAnnotationSet(version)

    def get_human_annotations(self):
        human_annotations = os.path.join(settings.ANNOTATION_DIR, "face_bbox_human.csv")
        return FaceBoundingBoxAnnotationSet(human_annotations)

    def get_versions(self):
        return [None] + FaceBoundingBoxAnnotationSet.get_versions()

    def _find_faces(self, img, method_idx=2):
        dets = []
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        if method_idx == 0:
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 2:
                gray = img
            faces = self.face_detector_opencv_haarcascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Convert to array of dlib rectangles
            for (x, y, w, h) in faces:
                dets.append(dlib.rectangle(x, y, x+w, y+h))

        elif method_idx == 1:
            dets = self.face_detector_dlib_hog(img, 1)
        else:
            cnn_dets = self.face_detector_dlib_cnn(img, 1)
            for cnn_d in cnn_dets:
                # different return type because it includes confidence, get the rect
                d = cnn_d.rect
                h = int((d.top() - d.bottom()) / 10.0)
                w = int((d.right() - d.left()) / 10.0)
                # cnn max margin detector seems to cut off the chin and this confuses landmark predictor,
                # expand height by 10%
                dets.append(dlib.rectangle(d.left() - w, d.top() + h,
                                            d.right() + w, d.bottom() - h))

        return [(float(d.left()) / img.shape[1], float(d.top()) / img.shape[0], float(d.right()) / img.shape[1], float(d.bottom()) / img.shape[0]) for d in dets]

    def annotate_image(self, img_id, img, prior_inputs):
        dets = self._find_faces(img, self.face_idx)
        self.annotation_set.set_annotation(img_id, dets)
        return {'face_bbox': dets}

    def show_annotation(self, img_id, img, annotations, index=None, zoom=False, target_size=None, colour=None):
        if colour is None:
            colour = self.colour

        if index is None:
            render_bounding_boxes(img, annotations['face_bbox'], colour)
        else:
            if zoom:
                from skimage.transform import rescale
                d = annotations['face_bbox'][index]
                w = img.shape[1]
                h = img.shape[0]
                d = (int(d[0] * w), int(d[1] * h), int(d[2] * w), int(d[3] * h))
                b = int((d[2] - d[0]) * 0.25)
                crop_box = (max(0, d[0] - b), max(0, d[1] - b), min(w, d[2] + b), min(h, d[3] + b))
                #crop_box = (max(0, d[0] - b), max(0, d[1] - b), min(w, d[2] + b), min(h, d[3] + b))
                img = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
                d = (d[0] - min(d[0], crop_box[0]), d[1] - min(d[1], crop_box[1]), d[2] - min(d[0], crop_box[0]), d[3] - min(d[1], crop_box[1]))
                rescale_factor = min(target_size[0] / float(img.shape[1]), target_size[1] / float(img.shape[0]))
                img = (255 * rescale(img, rescale_factor, anti_aliasing=True)).astype(np.uint8)

                cv2.rectangle(img,
                    ( int(d[0]*rescale_factor), int(d[1]*rescale_factor) ),
                    ( int(d[2]*rescale_factor), int(d[3]*rescale_factor) ),
                    colour, 2
                    )
            else:
                render_bounding_boxes(img, [annotations['face_bbox'][index]], colour)
        return img

    def count_annotations(self, annotations):
        if annotations is None:
            return 0
        return len(annotations.get('face_bbox', []))
