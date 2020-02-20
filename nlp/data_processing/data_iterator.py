import argparse
import itertools

import numpy as np
import random

from nlp.data_processing.data_processor import DataProcessor


class DataIterator:
    def __init__(self, voc, pairs):
        self.max_len = 10
        self.voc = voc
        self.pairs = pairs

    def indexes_from_sentence(self, sentence):
        return [self.voc.word2index[word] for word in sentence.split(' ')] + [self.voc.eos_token]

    def zero_padding(self, indexes):
        return list(itertools.zip_longest(*indexes, fillvalue=self.voc.pad_token))

    def binary_mask(self, padded_list):
        m = []
        for i, seq in enumerate(padded_list):
            m.append([])
            for token in seq:
                if token == self.voc.pad_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def process_inputs(self, inputs):
        indexes_batch = [self.indexes_from_sentence(sentence) for sentence in inputs]
        lengths = np.array([len(indexes) for indexes in indexes_batch])
        padded = np.array(self.zero_padding(indexes_batch), dtype=np.int64)
        return padded, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def process_outputs(self, outputs):
        indexes_batch = [self.indexes_from_sentence(sentence) for sentence in outputs]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padded_list = self.zero_padding(indexes_batch)
        mask = self.binary_mask(padded_list)
        mask = np.array(mask, dtype=np.bool)
        padded = np.array(padded_list, dtype=np.int64)
        return padded, mask, max_target_len

    # Returns all items for a given batch of pairs
    def process_data(self, pairs):
        pairs.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        inputs, outputs = [], []
        for pair in pairs:
            inputs.append(pair[0])
            outputs.append(pair[1])
        inp, lengths = self.process_inputs(inputs)
        output, mask, max_target_len = self.process_outputs(outputs)
        return inp, lengths, output, mask, max_target_len


def main():
    parser = argparse.ArgumentParser(description='data iterator to loop through data')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/cornell_movie_data/')
    args = parser.parse_args()

    data_processor = DataProcessor(args.data_dir)
    voc, pairs = data_processor.load_prepare_data()
    pairs = data_processor.trim_rare_words(voc, pairs)

    print("\nSample pairs:")
    for pair in pairs[:10]:
        print(pair)

    data_iter = DataIterator(voc, pairs)
    batch = [random.choice(pairs) for _ in range(2)]
    inp, lengths, output, mask, max_target_len = data_iter.process_data(batch)

    print("\ninput_variable:\n", inp)
    print("lengths:\n", lengths)
    print("target_variable:\n", output)
    print("mask:\n", mask)
    print("max_target_len:\n", max_target_len)


if __name__ == '__main__':
    main()
