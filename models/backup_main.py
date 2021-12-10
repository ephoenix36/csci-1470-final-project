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

from matplotlib import pyplot as plt
from PIL import Image

import numpy as np

from backup_model.attention import *
from backup_model.decoder import *
from backup_model.encoder import *
from backup_model.utils import *

RUN_ID = 1000 # MAKE SURE TO CHANGE THIS BEFORE RUNNING SO THAT PLOTS AND CAPTIONS ARE SAVED TO DIFFERENT FOLDER

def main():
    dataset, parameters, tokenizer, img_name_val, cap_val = preprocessing.preprocess()

    vocab_size = parameters[0]
    output_size = 20
    max_sentence_len = parameters[1]
    num_layers = 10
    padding_index = parameters[2]
    num_batches = parameters[3]
    embedding_dim = 256
    epochs = 15

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    image_feature_extraction_model = tf.keras.Model(
        image_model.input, image_model.layers[-1].output)

    encoder = Encoder(embedding_dim)
    decoder = Decoder(embedding_dim, output_size, vocab_size)
    
    print(len(dataset))

    losses = train(dataset, epochs, encoder, decoder, tokenizer, batch_size=64)

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    # plt.show()
    plt.savefig(f"../output_log/{RUN_ID}/loss_plot")
    
    captions = []

    num_images = 50
    for i in range(num_images):
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i]
                                for i in cap_val[rid] if i not in [0]])
        result, attention_plot = evaluate(image, max_sentence_len, encoder, decoder, tokenizer, image_feature_extraction_model)

        # print(f"\nImage ID: {rid}")
        # print('Real Caption:', real_caption)
        predicted_caption  =' '.join(result)
        # print('Predicted Caption:', predicted_caption)
        plot_attention(image, result, attention_plot, rid)
        
        text = "\n" + str(rid) + "\nReal Caption: " + real_caption + "\nPredicted Caption: " + predicted_caption
        captions.append(text)

    with open(f"../output_log/{RUN_ID}/results.csv", 'a') as results_append:
        np.savetxt(results_append, np.array(captions), delimiter=',', fmt='%s')

def train(dataset, epochs, encoder, decoder, tokenizer, batch_size):

    loss_plot = []

    for ep in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (input, target)) in enumerate(dataset):
            cumulative_batch_loss, loss = train_batch(
                input, target, encoder, decoder, tokenizer)
            total_loss += loss

            if batch % 50 == 0:
                average_batch_loss = cumulative_batch_loss.numpy() / \
                    int(target.shape[1])
                print(
                    f'Epoch {ep+1} Batch {batch} Loss {average_batch_loss:.4f}')

        loss_plot.append(total_loss / (batch * (ep + 1)))

        print(f'Epoch {ep + 1} Loss {total_loss / batch_size:.6f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    return loss_plot


# @tf.function
def train_batch(input, target, encoder: Encoder, decoder: Decoder, tokenizer):
    loss = 0

    optimizer = tf.keras.optimizers.Adam()

    hidden = decoder.reset_state(target.shape[0])

    decoder_input = tf.expand_dims(
        [tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(input)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(decoder_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            decoder_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    # print(encoder.trainable_variables, decoder.trainable_variables)
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    # trainable_variables = tf.concat([tf.constant(encoder.trainable_variables), tf.constant(decoder.trainable_variables)], axis=0)

    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def evaluate(image, max_length, encoder, decoder, tokenizer, feature_extractor):
    attention_features_shape = 64 
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    input_features = feature_extractor(temp_input)
    input_features = tf.reshape(
        input_features, (input_features.shape[0], -1, input_features.shape[3]))

    input_features = encoder(input_features)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    
    result = []
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         input_features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot, rid):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(np.ceil(len_result/3), 3)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../output_log/{RUN_ID}/figure{rid}")


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


if __name__ == '__main__':
    main()
