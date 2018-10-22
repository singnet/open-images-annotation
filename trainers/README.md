## Model trainers for the Open Images annotation project

Currently there are two trainers:

- dlib_face_detection_trainer - trains a CNN model compatible with the CNN model released by dlib.
- dlib_face_landmarks_trainer - trains a shape predictor model compatible with the dlib 68-pt shape_predictor

dlib's default CNN face detection model is pretty good and unhampered by annoying restrictions. The 68-pt face landmark
predictor available from dlib is [in a legal grey area](https://github.com/davisking/dlib-models), which is partly the inspiration for this project.

### Build instructions

From the directory containing this README:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Training a model

You need to convert one of the annotation CSVs to a dlib XML dataset.

There is a script that can help with this, generating either bounding boxes only or
bounding boxes and landmarks.

`open-images-annotation/scripts/convert_csv_to_dlib_xml.py --help`

It should work with either the landmark CSV or bounding box CSV and has options
to output matched OpenImages bounding boxes, and/or to force the bounding boxes to
be square... which seems to work better for dlib's non-max-supression loss. Having
too many output box layers (due to different aspect ratios and sizes) seems to
hinder convergence.

### Caveats

- The face detector training didn't converge using the full set of open images face bounding boxes,
  even if these are made square first. Further loss parameter exploration, or fine-tuning on a model trained on a
  smaller set of data, may fix this. One issue is the open images labels human faces in a wide array of
  orientations including when most of the face is occluded.
- The landmark trainer needs all images to be in memory to train, limit the number examples exported to
  the training XML file with the convert scripts `--max-count` parameter.
- The landmark trainer seems to return a loss of NaN if using more than ~5000 examples.
- Open-Images provides training, validation, and test splits. We have only be using splits of the training data so far.

Despite these caveats, the models work okay and allow us to bootstrap a labelling process
with a small number of human created predictions.

As an initial example we hope this will be built upon. Replacing dlib with a more capable object detector
and landmark predictor may be worth while, particularly to take advantage of the full set of landmark
annotations.