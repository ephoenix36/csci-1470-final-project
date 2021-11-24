from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds # @team: pip install this
from model import *
from util import *

@timing
def train(model, x_train, y_train):
    """
    """
    assert(x_train.shape[0] == y_train.shape[0])
    
    # TODO: lookup how to do batching with tfds
    
    # for start, end in zip(range(0, x_train.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, x_train.shape[0], model.batch_size)):
    
    #     with tf.GradientTape() as tape:
    #         predictions = model.call(batch_inputs, batch_labels[:, :-1])
    #         loss = model.loss_function(predictions, batch_labels[:, 1:], mask)
        
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #     # model.optimizer.apply_gradients(
    #     #     (grad, var) 
    #     #     for (grad, var) in zip(gradients, model.trainable_variables) 
    #     #     if grad is not None
    #     # )

    #     if start % (model.batch_size * 100) == 0:
    #         print("\tLoss on Training Batch {}: {}".format(start//model.batch_size, loss))
    #         # print(model.accuracy_function(predictions, batch_labels[:, 1:], mask).numpy())

@timing
def test(model, x_test, y_test):
    """
    """
    
    pass

    # total_loss = 0
    # correct_predictions = 0
    # total_output_words = 0
    # for start, end in zip(range(0, test_french.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, test_french.shape[0], model.batch_size)):
    #     batch_inputs = test_french[start:end]
    #     batch_labels = test_english[start:end]

    #     mask = np.where(batch_labels[:, 1:] != eng_padding_index, 1, 0)
    #     # mask = (batch_labels[:, 1:] != eng_padding_index)


    #     num_words = tf.reduce_sum(mask)
    #     total_output_words += num_words

    #     predictions = model.call(batch_inputs, batch_labels[:, :-1])
    #     loss = model.loss_function(predictions, batch_labels[:, 1:], mask)
    #     total_loss += loss
    #     num_correct_predictions = model.accuracy_function(
    #         predictions, batch_labels[:, 1:], mask).numpy() * num_words.numpy()
    #     correct_predictions += num_correct_predictions
    #     # print("\tLoss on Test Batch {}: {}".format(start//model.batch_size, loss))

    # # avg_loss = total_loss / total_output_words
    # # print("\tAverage Test Loss: {}".format(avg_loss))
    # perplexity = np.exp(np.cast['float32'](total_loss) / np.cast['float32'](total_output_words))
    # accuracy = correct_predictions / total_output_words
    # return perplexity, accuracy.numpy()


def main():

    # TODO: add preprocessing
    coco = tfds.object_detection.Coco
    (x_train, y_train), (x_test, y_test) = coco.load_data()

    model_args = []
    model = MeshedMemoryModel(*model_args)

    train(model, x_train, y_train)

    perplexity, accuracy = test(model, x_test, y_test)
    
    # TODO: figure out how to use tensorboard to visualize results
    
    pass


if __name__ == '__main__':
    main()
