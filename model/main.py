import preprocessing

from model.transformer.meshedModel import *


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
