import preprocessing
from model.transformer.model import MeshedMemoryModel


def main():
    vocab_size = 10
    max_sentence_len = 10
    num_layers = 10
    padding_index = 10
    output_size = 10

    model = MeshedMemoryModel(vocab_size, max_sentence_len, num_layers, padding_index, output_size)

    dataset = preprocessing.post_preprocess()


if __name__ == '__main__':
    main()
