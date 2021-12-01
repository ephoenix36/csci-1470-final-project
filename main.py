'''
USEFUL LINKS:

Paper: https://arxiv.org/pdf/1912.08226v2.pdf
Paper's Github: https://github.com/aimagelab/meshed-memory-transformer

Using tensorflow datasets: https://www.tensorflow.org/datasets/keras_example

LESS USEFUL LINKS:
Using coco dataset and tools: https://github.com/tylin/coco-caption
'''

from datetime import datetime
import numpy as np
import tensorflow as tf
from model.transformer.model import *
# from .transformer.model import *
from utils.utils import *

# pip install -U tensorflow_datasets
import tensorflow_datasets as tfds


@timing
def train(model, x_train: tf.data.Dataset, y_train: tf.data.Dataset, batch_size=100):
    """
    """
    x_train = x_train.batch(batch_size, drop_remainder=True)
    y_train = y_train.batch(batch_size, drop_remainder=True)

    # TODO: update batching to match preprocessing

    for start, end in zip(range(0, x_train.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, x_train.shape[0], model.batch_size)):
        batch_inputs = x_train[start:end]
        batch_labels = y_train[start:end]

        with tf.GradientTape() as tape:
            predictions, mask_encoder = model.call(
                batch_inputs, batch_labels[:, :-1])
            loss = model.loss_function(
                predictions, batch_labels[:, 1:], mask_encoder)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        # model.optimizer.apply_gradients(
        #     (grad, var)
        #     for (grad, var) in zip(gradients, model.trainable_variables)
        #     if grad is not None
        # )

        if start % (model.batch_size * 100) == 0:
            print("\tLoss on Training Batch {}: {}".format(
                start//model.batch_size, loss))


@timing
def test(model, x_test, y_test, padding_index):
    """
    """

    total_loss = 0
    correct_predictions = 0
    total_output_words = 0
    for start, end in zip(range(0, x_test.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, x_test.shape[0], model.batch_size)):
        batch_inputs = x_test[start:end]
        batch_labels = y_test[start:end]

        mask = np.where(batch_labels[:, 1:] != padding_index, 1, 0)

        num_words = tf.reduce_sum(mask)
        total_output_words += num_words

        predictions = model.call(batch_inputs, batch_labels[:, :-1])
        loss = model.loss_function(predictions, batch_labels[:, 1:], mask)
        total_loss += loss
        num_correct_predictions = model.accuracy_function(
            predictions, batch_labels[:, 1:], mask).numpy() * num_words.numpy()
        correct_predictions += num_correct_predictions
        # print("\tLoss on Test Batch {}: {}".format(start//model.batch_size, loss))

    # avg_loss = total_loss / total_output_words
    # print("\tAverage Test Loss: {}".format(avg_loss))
    perplexity = np.exp(np.cast['float32'](
        total_loss) / np.cast['float32'](total_output_words))
    accuracy = correct_predictions / total_output_words
    return perplexity, accuracy.numpy()


def main():

    # TODO: add preprocessing
    # (x_train, y_train), (x_test, y_test) = tfds.load(
    #     'coco_captions',
    #     split=['train', 'test'],
    #     shuffle_files=True,
    #     as_supervised=True,
    #     with_info=True,
    # )
    
    (x_test, y_test) = tfds.load(
        'coco_captions',
        split=['test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    for i in x_test.as_numpy_iterator():
        print(i)

    # model = MeshedMemoryModel()  # TODO: fill in args

    # train(model, x_train, y_train)

    # perplexity, accuracy = test(model, x_test, y_test)
    # print(f"Perplexity {perplexity}, Accuracy {accuracy}")

    # # TODO: figure out how to use tensorboard to visualize results

    # pass


if __name__ == '__main__':
    main()
