import os
import argparse

from nlp.data_processing.data_utils import normalize_str


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.index2word = {self.pad_token: "PAD", self.sos_token: "SOS", self.eos_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("\nKept words: {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_token: "PAD", self.sos_token: "SOS", self.eos_token: "EOS"}
        self.num_words = 3    # Count default tokens

        for word in keep_words:
            self.add_word(word)


class DataProcessor:
    def __init__(self, data_dir):
        self.corpus_name = "cornell_movie_dialogs"
        self.data_fn = os.path.join(data_dir, "formatted_movie_lines.txt")
        self.max_len = 10
        self.min_count = 3

    # Filter pairs using filterPair condition
    def filter_pairs(self, pairs):
        # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
        def filter_pair(pair):
            # Input sequences need to preserve the last word for EOS token
            return len(pair[0].split(' ')) < self.max_len and len(pair[1].split(' ')) < self.max_len
        return [pair for pair in pairs if filter_pair(pair)]

    # Read query/response pairs and return a voc object
    def read_vocs(self):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(self.data_fn, encoding='utf-8').read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalize_str(s) for s in line.split('\t')] for line in lines]
        voc = Voc(self.corpus_name)
        return voc, pairs

    # Using the functions defined above, return a populated voc object and pairs list
    def load_prepare_data(self):
        print("Start preparing training data ...")
        voc, pairs = self.read_vocs()
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("\nCounting words...")
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs

    def trim_rare_words(self, voc, pairs):
        voc.trim(self.min_count)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs))
        )
        return keep_pairs


def main():
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/cornell_movie_data/')
    args = parser.parse_args()

    data_processor = DataProcessor(args.data_dir)
    voc, pairs = data_processor.load_prepare_data()
    pairs = data_processor.trim_rare_words(voc, pairs)

    print("\nSample pairs:")
    for pair in pairs[:10]:
        print(pair)


if __name__ == '__main__':
    main()
