import os
import argparse
import glob

import tensorflow as tf


def get_dataset(tfrecords,
                batch_size,
                epochs,
                max_len=10,
                shuffle=False,
                fake_input=False):
    def parse_func(example_proto):
        feature_dict = {
            'input/sentence': tf.FixedLenFeature([], tf.string),
            'input/encoded': tf.FixedLenFeature([max_len], tf.int64),
            'input/length': tf.FixedLenFeature([], tf.int64),
            'label/sentence': tf.FixedLenFeature([], tf.string),
            'label/encoded': tf.FixedLenFeature([max_len], tf.int64),
            'label/length': tf.FixedLenFeature([], tf.int64)
        }
        parsed_feature = tf.parse_single_example(example_proto, feature_dict)

        features, labels = {}, {}
        for key, val in parsed_feature.items():
            if 'input' in key:
                features[key] = val
            elif 'label' in key:
                if 'encoded' in key:
                    input_target = tf.identity(val)
                    eos_token = tf.constant(1, dtype=tf.int64, shape=input_target.get_shape().as_list()[:-1] + [1])
                    input_target = tf.concat([eos_token, input_target], axis=-1)
                    input_target = tf.slice(input_target, [0], [max_len])
                    features['input/input_target'] = input_target
                labels[key] = val
        return features, labels

    files = tf.data.Dataset.list_files(tfrecords)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=256, count=epochs))
    else:
        dataset = dataset.repeat(epochs)
    dataset = dataset.map(parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)
    if fake_input:
        dataset = dataset.take(1).cache().repeat()
    return dataset


def data_input_fn(tfrecords,
                  batch_size,
                  epochs,
                  max_len=10,
                  voc_size=7826,
                  shuffle=False,
                  fake_input=False):
    def _input_fn():
        dataset = get_dataset(tfrecords,
                              batch_size,
                              epochs,
                              max_len,
                              shuffle,
                              fake_input)

        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next()

        input_sentence = next_batch[0]['input/encoded']
        input_target = next_batch[0]['input/input_target']
        label = next_batch[1]['label/encoded']

        # Pre-processing
        with tf.name_scope('preprocess_inputs'):
            input_sentence = tf.cast(input_sentence, dtype=tf.float32)
            input_target = tf.cast(input_target, dtype=tf.float32)

        with tf.name_scope('preprocess_labels'):
            one_hot_label = tf.one_hot(label, depth=voc_size, axis=-1)
            one_hot_label = tf.cast(one_hot_label, dtype=tf.float32)

        features, labels = {}, {}
        features['input_sentence'] = input_sentence
        features['input_target'] = input_target
        labels['label'] = label
        labels['one_hot_label'] = one_hot_label
        return features, labels
    return _input_fn


def main():
    parser = argparse.ArgumentParser(description='visualize input pipeline')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/cornell_movie_data/tfrecords/')
    parser.add_argument('--max-len', '-l', type=int, default=10)
    parser.add_argument('--voc-size', '-s', type=int, default=7826)
    parser.add_argument('--data-type', '-t', type=str, default='train')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tfrecords = glob.glob('{}{}/*.tfrecord'.format(args.data_dir, args.data_type))
    dataset = get_dataset(tfrecords,
                          batch_size=32,
                          epochs=1,
                          max_len=args.max_len,
                          shuffle=False,
                          fake_input=False)
    print('\nDataset out types {}'.format(dataset.output_types))

    batch = dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    try:
        batch_nb = 0
        while True:
            data = sess.run(batch)
            batch_nb += 1

            input_string = data[0]['input/sentence']
            input_encoded = data[0]['input/encoded']
            input_len = data[0]['input/length']
            label_string = data[1]['label/sentence']
            label_encoded = data[1]['label/encoded']
            label_len = data[1]['label/length']

            print('\nBatch nb {}'.format(batch_nb))
            for i in range(len(input_string)):
                print("\nInputs:")
                print(input_len, input_string, input_encoded)
                print("\nLabels:")
                print(label_len, label_string, label_encoded)

    except tf.errors.OutOfRangeError:
        pass


if __name__ == '__main__':
    main()
