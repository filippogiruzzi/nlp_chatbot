import os
import argparse

import csv
import re
import codecs
from io import open


class DataFormatter:
    def __init__(self, data_dir):
        self.convs_fn = os.path.join(data_dir, "movie_conversations.txt")
        self.lines_fn = os.path.join(data_dir, "movie_lines.txt")
        self.out_fn = os.path.join(data_dir, "formatted_movie_lines.txt")
        self.delimiter = str(codecs.decode('\t', "unicode_escape"))
        self.lines_fields = ["line_id", "char_id", "movie_id", "char", "text"]
        self.convs_fields = ["char1_id", "char2_id", "movie_id", "utterance_ids"]
        self.lines = {}
        self.convs = []

    # Splits each line of the file into a dictionary of fields
    def load_lines(self):
        print("\nProcessing corpus...")
        with open(self.lines_fn, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                line_obj = {}
                for i, field in enumerate(self.lines_fields):
                    line_obj[field] = values[i]
                self.lines[line_obj["line_id"]] = line_obj

    # Groups fields of lines from `load_lines` into conversations based on *movie_conversations.txt*
    def load_convs(self):
        print("\nLoading conversations...")
        with open(self.convs_fn, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                conv_obj = {}
                for i, field in enumerate(self.convs_fields):
                    conv_obj[field] = values[i]
                # Convert string to list (conv_obj["utterance_ids"] == "['L598485', 'L598486', ...]")
                utterance_id_pattern = re.compile('L[0-9]+')
                line_ids = utterance_id_pattern.findall(conv_obj["utterance_ids"])
                # Reassemble lines
                conv_obj["lines"] = []
                for line_id in line_ids:
                    conv_obj["lines"].append(self.lines[line_id])
                self.convs.append(conv_obj)

    # Extracts pairs of sentences from conversations
    def extract_sentence_pairs(self):
        qa_pairs = []
        for conv in self.convs:
            # Iterate over all the lines of the conversation
            for i in range(len(conv["lines"]) - 1):  # We ignore the last line (no answer for it)
                input_line = conv["lines"][i]["text"].strip()
                target_line = conv["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if input_line and target_line:
                    qa_pairs.append([input_line, target_line])
        return qa_pairs

    # Write new csv file
    def write_formatted_file(self):
        print("\nWriting newly formatted file...")
        with open(self.out_fn, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.extract_sentence_pairs():
                writer.writerow(pair)

    def print_lines(self, n=10):
        print("\nSample from formatted file:\n")
        try:
            with open(self.out_fn, 'rb') as f:
                lines = f.readlines()
            for line in lines[:n]:
                print(line)
        except FileNotFoundError:
            print("Write formatted file first !!!")


def main():
    parser = argparse.ArgumentParser(description="data loader")
    parser.add_argument("--data-dir", type=str, default="/home/filippo/datasets/cornell_movie_data/")
    args = parser.parse_args()

    data_formatter = DataFormatter(args.data_dir)
    data_formatter.load_lines()
    data_formatter.load_convs()
    data_formatter.write_formatted_file()
    data_formatter.print_lines()


if __name__ == '__main__':
    main()
