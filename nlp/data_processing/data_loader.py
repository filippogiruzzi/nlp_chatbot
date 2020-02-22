import argparse

import numpy as np

from nlp.data_processing.data_processor import DataProcessor


class DataLoader:
    def __init__(self, voc, pairs, max_len=10, split=0.7):
        self.max_len = max_len
        self.split = split
        self.random_seed = 123456
        self.voc = voc
        self.pairs = pairs

    def indexes_from_sentence(self, sentence):
        return [self.voc.word2index[word] for word in sentence.split(' ')] + [self.voc.eos_token]

    def zero_padding(self, indexes):
        padded = np.pad(indexes, (0, self.max_len - len(indexes)), 'constant', constant_values=self.voc.pad_token)
        return padded

    def get_split(self):
        np.random.seed(self.random_seed)
        np.random.shuffle(self.pairs)
        n = len(self.pairs)
        n_train, n_val = int(self.split * n), int((1 - self.split) * n / 2)
        train, val, test = self.pairs[:n_train], self.pairs[n_train:n_train + n_val], self.pairs[n_train + n_val:]
        print("\nDataset split: {} | {} | {}".format(len(train), len(val), len(test)))
        print("Total {} sentences".format(len(train) + len(val) + len(test)))
        return train, val, test

    def process_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        padded = self.zero_padding(indexes)
        return {'sentence': sentence, 'inputs': padded, 'length': len(indexes)}

    def data_iter(self, pairs):
        skipped = 0
        for pair in pairs:
            if len(pair) > 2:
                skipped += 1
                continue
            inputs, labels = pair
            input_dict = self.process_sentence(inputs)
            label_dict = self.process_sentence(labels)
            yield {'inputs': input_dict, 'labels': label_dict}


def main():
    parser = argparse.ArgumentParser(description='data iterator to loop through data')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/cornell_movie_data/')
    args = parser.parse_args()

    data_processor = DataProcessor(args.data_dir)
    voc, pairs = data_processor.load_prepare_data()
    pairs = data_processor.trim_rare_words(voc, pairs)

    print("\nSample pairs:")
    for pair in pairs[:10]:
        print(pair)

    data_loader = DataLoader(voc, pairs)
    train, val, test = data_loader.get_split()
    data_iter = data_loader.data_iter(pairs=train)
    for data in data_iter:
        print()
        print(data['inputs'])
        print(data['labels'])


if __name__ == '__main__':
    main()
