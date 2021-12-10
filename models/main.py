'''
USEFUL LINKS:

Paper: https://arxiv.org/pdf/1912.08226v2.pdf
Paper's Github: https://github.com/aimagelab/meshed-memory-transformer

Using tensorflow datasets: https://www.tensorflow.org/datasets/keras_example

LESS USEFUL LINKS:
Using coco dataset and tools: https://github.com/tylin/coco-caption
'''

import preprocessing
import time

# from transformer.meshedModel import *
from bahduana_model import *
from model import *


def main():
    dataset, parameters, tokenizer = preprocessing.preprocess()

    vocab_size = parameters[0]
    max_sentence_len = parameters[1]
    num_layers = 10
    padding_index = parameters[2]
    output_size = 10
    num_steps = parameters[3]

    # model = MeshedMemoryModel(vocab_size, max_sentence_len, num_layers, padding_index, output_size)

    encoder = Encoder(100)
    decoder = Decoder(100, output_size, vocab_size)
    # decoder = MeshedDecoder(vocab_size, max_sentence_len, num_layers, padding_index, output_size)
    # encoder = CNN_Encoder(100)
    # decoder = RNN_Decoder(100, output_size, vocab_size)
    
    # train(dataset, 10, model, tokenizer, loss_function)
    train(dataset, num_steps, encoder, decoder, tokenizer)

# def train(dataset, num_steps, model, tokenizer, loss_function):

def train(dataset, num_steps, encoder, decoder, tokenizer):

    EPOCHS = 20
    loss_plot = []

    # for epoch in range(start_epoch, EPOCHS):
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            # img_tensor = tf.expand_dims(img_tensor, 0)
            # target = tf.expand_dims(target, 0)
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer)
            total_loss += t_loss
            print("completed train step")
            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        # if epoch % 5 == 0:
        # ckpt_manager.save()

        print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')


@tf.function
def train_step(img_tensor, target, encoder, decoder, tokenizer):
# C:/Users/ephoe/Desktop/CSCI/CS1470/meshed-memory-transformer/environment.yml
    loss = 0
    optimizer = tf.keras.optimizers.Adam()
    # print("HEREEEEEEE")
    # print(model.encoder.layers[0].attention.mk.shape)
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = tf.zeros((target.shape[0], 10))

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        # decoder_output, mask_encoder = model.call(img_tensor, hidden)
        # encoder_output, mask_encoder = encoder(img_tensor)
        
        # loss += model.loss_function(decoder_output, target, mask_encoder)
        
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):

            # passing the features through the decoder
            # predictions = decoder(dec_input, encoder_output, mask_encoder)
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    # trainable_variables = tf.concat([encoder.trainable_variables, decoder.trainable_variables], axis=0)
    # trainable_variables = model.trainable_variables
    
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


if __name__ == '__main__':
    main()
