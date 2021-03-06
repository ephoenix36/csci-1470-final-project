import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import h5py
from PIL import Image
from skimage import io
import fiftyone.zoo as foz


def download_data():

    # downloads the fiftyone dataset onto your computer in "fiftyone folder"
    foz.load_zoo_dataset(
        "coco-2014",
        split="train",
        max_samples=85000,
        shuffle=False,
    )


def get_subimage_label_dicts(file):

    # opens the json as a python dictionary
    with open(file, 'r') as f:
        train_dict = json.load(f)

    # builds dictionary that turns category ids into labels
    category_dict = {}
    for element in train_dict["categories"]:
        category_dict[str(element["id"])] = element["name"]

    # build a dictionaries with conversions from image_id into bboxes and labels
    bbox_dict = {}
    label_dict = {}
    for element in train_dict["annotations"]:

        image_id = str(element["image_id"])
        bbox = element["bbox"]
        label = category_dict[str(element["category_id"])]

        if image_id not in bbox_dict:
            bbox_dict[image_id] = [bbox]
            label_dict[image_id] = [label]
        else:
            bbox_dict[image_id].append(bbox)
            label_dict[image_id].append(label)

    return bbox_dict, label_dict


def build_dictionaries(file):
    
    # opens the json as a python dictionary
    with open(file, 'r') as f:
        train_dict = json.load(f)

    # builds the image dictionary
    image_dict = train_dict["images"]
    
    # builds the caption dictionary
    caption_dict = {}
    for element in train_dict["annotations"]:
        image_id = str(element["image_id"])
        if image_id not in caption_dict:
            caption_dict[image_id] = [element["caption"]]
        else:
            caption_dict[image_id].append(element["caption"])

    # return dictionaries
    return image_dict, caption_dict


def get_image_batch(image_dict, bbox_dict, image_directory):

    image_batch = []
    for i in range(0,100):

        # get the big image
        location = image_directory + image_dict[i]["file_name"]
        big_image = np.asarray(Image.open(location))

        # get the bboxes for the image (if they exist)
        key = str(image_dict[i]["id"])
        if key in bbox_dict:
            bboxes = bbox_dict[key]
        else:
            bboxes = [[0,0,big_image.shape[0],big_image.shape[1]]]

        # apply bboxes to the image to generate subimage_list
        subimage_list = []
        for i in range(len(bboxes)):
            xmin, ymin, width, height = bboxes[i]
            subimage = big_image[int(ymin):int(ymin+height),int(xmin):int(xmin+width),:]
            subimage_list.append(subimage)
        image_batch.append(subimage_list)
    
    # returns the batch
    return image_batch
	

if __name__ == "__main__":

    # download the data if not already done
    # download_data()

    # build the dictionaries
    print("preprocessing started")
    print("Building dictionaries...")
    bbox_dict, label_dict = get_subimage_label_dicts("../../coco-2014/train/labels.json")
    image_dict, caption_dict = build_dictionaries("../../coco-2014/raw/captions_train2014.json")
    print("Built dictionaries")

    # find a batch of images and captions
    num_images = len(image_dict)
    batch_size = 100
    for i in range(0, num_images, batch_size):
        image_batch = get_image_batch(image_dict,bbox_dict,"../../coco-2014/train/data/")
        print("Batch complete, image number", i)
        break