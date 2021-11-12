from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
import sys
import random


def train(model, train_french, train_english, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """
    for start, end in zip(range(0, train_french.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, train_french.shape[0], model.batch_size)):
        batch_inputs = train_french[start:end]
        batch_labels = train_english[start:end]

        mask = np.where(batch_labels[:, 1:] != eng_padding_index, 1, 0)
        # mask = (batch_labels[:, 1:] != eng_padding_index)

        
        # print("Calling Transformer Model")
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs, batch_labels[:, :-1])
            loss = model.loss_function(predictions, batch_labels[:, 1:], mask)
        # print("Done Calling Transformer Model")
        
        gradients = tape.gradient(loss, model.trainable_variables)
        # model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.optimizer.apply_gradients(
            (grad, var) 
            for (grad, var) in zip(gradients, model.trainable_variables) 
            if grad is not None
        )

        if start % (model.batch_size * 100) == 0:
            print("\tLoss on Training Batch {}: {}".format(start//model.batch_size, loss))
            # print(model.accuracy_function(predictions, batch_labels[:, 1:], mask).numpy())

    pass


@av.test_func
def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
    e.g. (my_perplexity, my_accuracy)
    """

    total_loss = 0
    correct_predictions = 0
    total_output_words = 0
    for start, end in zip(range(0, test_french.shape[0] - model.batch_size, model.batch_size), range(model.batch_size, test_french.shape[0], model.batch_size)):
        batch_inputs = test_french[start:end]
        batch_labels = test_english[start:end]

        mask = np.where(batch_labels[:, 1:] != eng_padding_index, 1, 0)
        # mask = (batch_labels[:, 1:] != eng_padding_index)


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
    perplexity = np.exp(np.cast['float32'](total_loss) / np.cast['float32'](total_output_words))
    accuracy = correct_predictions / total_output_words
    return perplexity, accuracy.numpy()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()

    # Change this to "True" to turn on the attention matrix visualization.
    # You should turn this on once you feel your code is working.
    # Note that it is designed to work with transformers that have single attention heads.
    if sys.argv[1] == "TRANSFORMER":
        av.setup_visualization(enable=False)

    print("Running preprocessing...")
    train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = \
        get_data('C:/Users/ephoe/Desktop/CSCI/CS1470/hw4-seq2seq-ephoenix36/data/fls.txt', 'C:/Users/ephoe/Desktop/CSCI/CS1470/hw4-seq2seq-ephoenix36/data/els.txt',
                 'C:/Users/ephoe/Desktop/CSCI/CS1470/hw4-seq2seq-ephoenix36/data/flt.txt', 'C:/Users/ephoe/Desktop/CSCI/CS1470/hw4-seq2seq-ephoenix36/data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab),
                  ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args)

    print("Model:", sys.argv[1])

    # TODO:
    # Train and Test Model for 1 epoch.

    t = datetime.now()
    print("Training Model...")
    train(model, train_french, train_english, eng_padding_index)
    print("Model Trained ({})".format(datetime.now()-t))

    t = datetime.now()
    print("Testing Model ...")
    perplexity, accuracy = test(
        model, test_french, test_english, eng_padding_index)
    print("\tTest Perplexity: ", perplexity)
    print("\tTest Accuracy: ", accuracy)
    print("Model Tested ({})".format(datetime.now()-t))

    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    av.show_atten_heatmap()
    pass


if __name__ == '__main__':
    main()
