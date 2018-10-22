class Annotator(object):

    def __init__(self):
        self.requires = []
        self.provides = []

        # Maximum annotations to make before flush annotations should
        # be called.
        self.flush_interval = 100
        self.annotation_set = None

    def get_versions(self):
        """ Get all valid version strings in reverse-chronological order
        
        Index 0 is the current version that is used when annotate_image is called
        "human" is the version that includes annotations made by humans, or verified as Good.
        """
        return [None]

    def annotate_image(self, img_id, img, prior_inputs):
        """ Take in an image and generate new annotations. """
        raise NotImplementedError

    def show_annotation(self, img_id, img, annotations, index=None):
        """ Render annotation onto img """
        raise NotImplementedError

    def get_trainer(self):
        return AnnotationTrainer()

    def get_annotation_set(self):
        return AnnotationSet()

    def check_prior_inputs_ok(self, prior_inputs):
        """ Ensure any upstream annotations needed are provided """
        for input_req in self.requires:
            if input_req not in prior_inputs.keys():
                return False
        return True

    def flush_annotation_set(self):
        """ Flush any cached annotations to persistent storage.
        
        To match open images annotations, we should try to use CSVs where possible,
        but for segmentation maps and other dense annotations we will need binary
        formats.
        """
        if self.annotation_set:
            self.annotation_set.flush()


class AnnotationSet(object):

    def __init__(self):
        pass

    def get_version_name(self):
        raise NotImplementedError

    def get_annotation(self, img_id, version=None, index=None):
        """ Get an annotation.
        
        If version is None, return the latest.
        If version is a valid version string, attempt to load that from persistent storage. 

        Returns None if annotation is missing for combination of img_id and version
        """
        raise NotImplementedError

    def set_annotation(self, img_id, annotations):
        raise NotImplementedError

    # Consider refactoring these into a separate class that manages UI
    def rate_annotation(self, img_id, rating, version=None, index=None):
        """ Rate annotation 1/2/3 - Good/Ok/Bad """
        raise NotImplementedError

    def edit_annotation(self, img_id, img, annotations, index, window_context):
        """ Editing callback on window_context (unsure this is the right abstraction) """
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError
    

class AnnotationTrainer(object):
    def train(self, training_annotations):
        """ Train a new model with given annotations """
        raise NotImplementedError

