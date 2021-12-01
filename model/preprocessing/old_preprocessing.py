import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from skimage import io


def build_dictionaries(file):
    
    # opens the json as a python dictionary
    with open(file, 'r') as file:
        train_dict = json.load(file)

    # splits the training dictionary into images and captions
    image_dict = train_dict["images"]
    caption_dict = {}
    for element in train_dict["annotations"]:
        id = str(element["image_id"])
        if id not in caption_dict:
            caption_dict[id] = [element["caption"]]
        else:
            caption_dict[id].append(element["caption"])

    # return dictionaries
    return image_dict, caption_dict


def get_data_batch(image_dict, caption_dict, start_index, end_index):

    # finds a batch of images and captions by looking up via url online
    urls = []
    caption_batch = []
    for i in range(start_index,end_index):
        urls.append("http://images.cocodataset.org/train2014/" + image_dict[i]["file_name"])
        caption_batch.append(caption_dict[str(image_dict[i]["id"])])
    image_batch = list(map(io.imread, urls))

    # returns the batches
    return image_batch, caption_batch
	

if __name__ == "__main__":

    # build the dictionaries
    image_dict, caption_dict = build_dictionaries('../../data/captions_train2014.json')
    print("Built dictionaries")

    # find a batch of images and captions
    num_images = len(image_dict)
    batch_size = 100
    for i in range(0, num_images, batch_size):
        image_batch, caption_batch = get_data_batch(image_dict,caption_dict, i, i+batch_size)
        print("Batch complete, image number", i)