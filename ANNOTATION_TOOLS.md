## Web annotation research

We provide is a simple QT app for visualising and reviewing annotations.

This does not support multiple users making a concerted annotation effort.
Having a web interface and supporting backend to automatically retrain models would be ideal.

Here is a brief list of some existing open source annotation systems, in case
they can be repurposed, built upon, or merely used for inspiration.

### js-segment-annotator

Uses superpixels to annotate, BSD license:
https://github.com/kyamagu/js-segment-annotator
http://kyamagu.github.io/js-segment-annotator/?view=edit&id=1

### image-labelling-tool

More complete example, MIT license, includes python flask app
https://github.com/yuyu2172/image-labelling-tool
fork of: https://bitbucket.org/ueacomputervision/image-labelling-tool/src

### LabelMeAnnotationTool

Advanced annotation tool:
https://github.com/CSAILVision/LabelMeAnnotationTool
has mturk integration
http://labelme2.csail.mit.edu/Release3.0/browserTools/php/mechanical_turk.php

It'd be worth learning at least some info about mturk integration.

### CVAT

CVAT is completely re-designed and re-implemented version of Video Annotation Tool
from Irvine, California tool. It is free, online, interactive video and
image annotation tool for computer vision. It is being used by our team to
annotate million of objects with different properties. Many UI and UX
decisions are based on feedbacks from professional data annotation team.
https://github.com/opencv/cvat

Has a UI, but doesn't use superpixels and instead uses tedious polygon drawing.


### VGG web annotator

http://www.robots.ox.ac.uk/~vgg/software/via/via_face_demo.html

### Microsoft CNTK has iterative bootstrapping model idea too

MS talks bout an iterative model with CNTK
https://www.microsoft.com/developerblog/2017/04/10/end-end-object-detection-box/

### PixelAnnotationTool

Alternative way to speed up pixel labelling UI
opencv watershed algorithm, qt app
https://github.com/abreheret/PixelAnnotationTool

### Other references

Lots of examples here...
https://www.quora.com/What-is-the-best-image-labeling-tool-for-object-detection
