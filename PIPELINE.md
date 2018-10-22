## Pipeline

The pipeline involves creating a small human labelled dataset, training a model,
applying the model to the full dataset, then reviewing the annotations and
marking them as good/bad.

Humans can also spend additional effort to correct the annotation.

## Dependencies between tasks

One issue with training a model is that it can have dependencies on previous labelling tasks.

Taking the face landmark tracker for example. This model is sensitive to the
source of bounding box, ideally we could use any bounding box, either from
Open Images, a face detector, or a more general object detection and
localisation algorithm.

So when training a new landmark detection model, which source should we use?
- use them all and hope this provides robustness in the landmark tracking,
  but this does not work so well for the regression tree algorithm we use.
- train a landmark detection model for each source of bounding boxes, this
  would provide multiple correct machine-generated annotations for human
  annotators to validate. Instead of rating or fixing, they'd get the hopefully
  easier task of just picking which is better.

This is compounded by having both 5-point and 68-point models, and having to
merge subsequent human annotations that may have different statistical
properties. However this difference in human annotations is something any
annotation effort needs to account for.

## Complete pipeline description

To be completed.

### Face Detection

### Face Landmark Prediction