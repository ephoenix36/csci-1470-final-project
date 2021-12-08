import preprocessing
import time

from transformer.meshedModel import *


def main():

    dataset, parameters = preprocessing.post_preprocess()

    vocab_size = parameters[0]
    max_sentence_len = parameters[1]
    num_layers = 10
    padding_index = parameters[2]
    output_size = 10

    model = MeshedMemoryModel(vocab_size, max_sentence_len, num_layers, padding_index, output_size)

    


if __name__ == '__main__':
    main()



def train(dataset, num_steps, encoder, decoder, tokenizer, loss_function):

    EPOCHS = 20
    loss_plot = []

    # for epoch in range(start_epoch, EPOCHS):
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

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
def train_step(img_tensor, target):

    loss = 0
    optimizer = tf.keras.optimizers.Adam()

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):

            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss