import io

import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def preprocess():
    semi_path = os.path.abspath('')
    preprocessing_path = "/data/preprocess_data.npy"
    dataset_path = "/data/dataset"
    tokenizer_path = "/data/tokenizer.json"
    has_data = os.path.exists(semi_path + dataset_path)
    has_preprocessing_params = os.path.exists(semi_path + preprocessing_path)
    has_tokenizer = os.path.exists(semi_path + tokenizer_path)
    if not has_data or not has_preprocessing_params or not has_tokenizer:
        if not has_data:
            print("No Dataset")
        if not has_preprocessing_params:
            print("No Params")
        if not has_tokenizer:
            print("No Tokenizer")
        get_and_store_data()

    adjusted_dataset, return_values, loaded_tokenizer, img_name_val, cap_val = post_get_data()

    return adjusted_dataset, return_values, loaded_tokenizer, img_name_val, cap_val


def get_and_store_data():
    # Download caption annotation files
    annotation_folder = '/data/annotations/'
    if not os.path.exists(os.path.abspath('') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(''),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/data/annotations/captions_train2014.json'
        os.remove(annotation_zip)
    else:
        annotation_file = os.path.abspath('') + '/data/annotations/captions_train2014.json'

    # Download image files
    image_folder = '/data/train2014/'
    if not os.path.exists(os.path.abspath('') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath(''),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('') + image_folder

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    # lead to 30,000 examples.
    train_image_paths = image_paths[:10000] # TODO: cahnge back to 10000
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    print(train_captions[0])
    Image.open(img_name_vector[0])

    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path_to_dataset in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path_to_dataset):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # Choose the top 5000 words from the vocabulary
    top_k = 5000 # TODO: change back to 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    pad_index = 0

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_cap_length = cap_vector.shape[1]

    # Calculates the max_length, which is used to store the attention weights
    max_length = max(len(t) for t in train_seqs)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)
    
    np.save(os.path.abspath('.') + '/data/img_name_val', img_name_val)
    np.save(os.path.abspath('.') + '/data/cap_val', cap_val)

    # Feel free to change these parameters according to your system's configuration

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    main_parameters = np.array([vocab_size, max_cap_length, pad_index, num_steps])

    path_to_dataset = os.path.abspath('.') + '/data/dataset/'
    tf.data.experimental.save(dataset, path_to_dataset)
    np.save(os.path.abspath('.') + '/data/preprocess_data', main_parameters)

    tokenizer_json = tokenizer.to_json()
    with io.open('data/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    pass


def post_get_data():
    # retrieve dataset from file location
    path = os.path.abspath('') + '/data/dataset/'
    spec = (
    tf.TensorSpec(shape=None, dtype=tf.float32, name=None), tf.TensorSpec(shape=None, dtype=tf.int32, name=None))
    dataset = tf.data.experimental.load(path, spec)
    return_values = np.load(os.path.abspath('.') + '/data/preprocess_data.npy')

    # shuffle and batch dataset
    BATCH_SIZE = 64 # HARDCODED IN BACKUP MODEL
    BUFFER_SIZE = 1000
    adjusted_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    adjusted_dataset = adjusted_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    with open('data/tokenizer.json') as f:
        data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)
    
    img_name_val = np.load(os.path.abspath('.') + '/data/img_name_val.npy')
    cap_val = np.load(os.path.abspath('.') + '/data/cap_val.npy')

    return adjusted_dataset, return_values, loaded_tokenizer, img_name_val, cap_val


if __name__ == '__main__':
    # dataset, return_values = preprocess_main()
    # path = os.path.abspath('.') + '/dataset/'
    # tf.data.experimental.save(dataset, path)
    # np.save(os.path.abspath('.') + '/preprocess_data', return_values)

    # print("load start")
    # post_get_data()
    # print("end load")


    preprocess()
