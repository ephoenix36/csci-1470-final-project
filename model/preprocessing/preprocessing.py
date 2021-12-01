import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from skimage import io
# from pycocotools.coco import COCO as cocoDataset
import fiftyone as fo
import fiftyone.zoo as foz

def get_data(training_file, testing_file):
    """
    gets the training data from the data folder in the json file
    """

    # fiftyone stuffff

    #
    # Load 50 random samples from the training split
    #
    # Only the required images will be downloaded (if necessary).
    # By default, only detections are loaded
    #

    dataset = foz.load_zoo_dataset(
        "coco-2014",
        split="train",
        max_samples=50,
        shuffle=True,
    )

    # session = fo.launch_app(dataset, port=5051)


    return None
	


if __name__ == "__main__":
    get_data('../data/captions_train2014.json','../data/captions_val2014.json')
