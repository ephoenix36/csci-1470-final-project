import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from skimage import io
from pycocotools.coco import COCO as cocoDataset
import fiftyone as fo
import fiftyone.zoo as foz

def get_data(training_file, testing_file):
    """
    gets the training data from the data folder in the json file
    """

    data_ids = {}
    # data_ids['train_ids'] = np.load("../data/coco_train_ids.npy")
    # data_ids['test_ids'] = np.load("../data/coco_test_ids.npy")
    # data_ids['dev_ids'] = np.load("../data/coco_dev_ids.npy")
    # data_ids['restval_ids'] = np.load("../data/coco_restval_ids.npy")

    initialized_coco = cocoDataset(annotation_file="../data/captions_train2014.json")

    # print(initialized_coco.imgs)
    # print(initialized_coco.anns)

    data_ids['train_ids'] = list(initialized_coco.imgs.keys())
    data_ids['test_ids'] = np.load("../data/coco_test_ids.npy")
    data_ids['dev_ids'] = np.load("../data/coco_dev_ids.npy")
    data_ids['restval_ids'] = np.load("../data/coco_restval_ids.npy")

    print(data_ids['train_ids'])

    a = initialized_coco.download(tarDir="/Users/jacobaxel/deepLearning/projects/csci-1470-final-project/data", imgIds=[data_ids['train_ids'][0]])
    # print(a)
    # one_image = initialized_coco.loadImgs([3, 6])
    # print(one_image)


    # fiftyone stuffff

    #
    # Load 50 random samples from the validation split
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
