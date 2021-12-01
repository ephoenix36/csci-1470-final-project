import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from skimage import io

def get_data(training_file, testing_file):
    """
    gets the training data from the data folder in the json file
    """
    
    # opens the json as a python dictionary
    with open(training_file, 'r') as file:
        train_dict = json.load(file)

    # these are the different dictionaries inside the dataset
    # info
    # images
    # licenses
    # annotations

    # splits the training dictionary into images and captions
    image_dict = train_dict["images"]
    annotations_dict = train_dict["annotations"]
    
    # finds the number of images/captions
    num_train_inputs = len(image_dict)

    # this prints out the image indices that are accessible
    with open('good_indices.txt', 'a') as f:
        for i in range(num_train_inputs):
            if i % 200 == 0:
                print(i)
            image_url = image_dict[i]["flickr_url"]
            try:
                array = io.imread(image_url)
                f.write(str(i))
                f.write("\n")
            except:
                pass
    
    # THIS IS TRYING TO LOAD THE DATASET FROM TENSORFLOW DATASETS
    # ds = tfds.load('coco_captions', split='train')
    # for example in ds:
    #     print(example)


    train_images = 0
    train_captions = 0
    test_images = 0
    test_captions = 0

    return train_images, train_captions, test_images, test_captions
	


if __name__ == "__main__":
    get_data('../data/captions_train2014.json','../data/captions_val2014.json')