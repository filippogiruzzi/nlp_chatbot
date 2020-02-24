import os
import multiprocessing
from absl import flags, app
from copy import deepcopy

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json

from nlp.data_processing.data_processor import DataProcessor
from nlp.data_processing.data_loader import DataLoader


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/cornell_movie_data/',
                    'data directory path')
flags.DEFINE_float('data_split',
                   '0.7',
                   'dataset split')
flags.DEFINE_integer('max_len',
                     10,
                     'sentence max length')
flags.DEFINE_integer('num_shards',
                     256,
                     'number of tfrecord files')
flags.DEFINE_boolean('debug',
                     False,
                     'debug for a few samples')
flags.DEFINE_string('data_type',
                    'trainvaltest',
                    'data types to write into tfrecords')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(inputs, labels):
    feature_dict = {
        'input/sentence': bytes_feature(inputs['sentence'].encode()),
        'input/encoded': int64_list_feature(inputs['inputs'].tolist()),
        'input/length': int64_feature(inputs['length']),
        'label/sentence': bytes_feature(labels['sentence'].encode()),
        'label/encoded': int64_list_feature(labels['inputs'].tolist()),
        'label/length': int64_feature(labels['length'])
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def pool_create_tf_example(args):
    return create_tf_example(*args)


def write_tfrecords(path, dataiter, num_shards=256, nmax=-1):
    writers = [
        tf.python_io.TFRecordWriter('{}{:05d}_{:05d}.tfrecord'.format(path, i, num_shards)) for i in range(num_shards)
    ]
    print('\nWriting to output path: {}'.format(path))
    pool = multiprocessing.Pool()
    counter = 0
    for i, tf_example in tqdm(enumerate(pool.imap(pool_create_tf_example, [(deepcopy(data['inputs']),
                                                                            deepcopy(data['labels'])
                                                                            ) for data in dataiter]))):
        if tf_example is not None:
            writers[i % num_shards].write(tf_example.SerializeToString())
            counter += 1
        if 0 < nmax < i:
            break
    pool.close()
    for writer in writers:
        writer.close()
    print('Recorded {} signals'.format(counter))


def create_tfrecords(data_dir,
                     max_len=10,
                     split=0.7,
                     num_shards=256,
                     debug=False,
                     data_type='trainvaltest'):
    np.random.seed(0)

    output_path = os.path.join(data_dir, 'tfrecords/')
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    data_processor = DataProcessor(data_dir, max_len=max_len)
    voc, pairs = data_processor.load_prepare_data()
    pairs = data_processor.trim_rare_words(voc, pairs)
    data_loader = DataLoader(voc, pairs, max_len=max_len, split=split)

    voc_data = {'index2word': voc.index2word, 'word2index': voc.word2index, 'word2count': voc.word2count}
    voc_fp = os.path.join(output_path, 'voc.json')
    with open(voc_fp, 'w') as f:
        print('\nDumped vocabulary data')
        json.dump(voc_data, f)

    train, val, test = data_loader.get_split()

    train_it = data_loader.data_iter(pairs=train)
    val_it = data_loader.data_iter(pairs=val)
    test_it = data_loader.data_iter(pairs=test)

    # Write data to tfrecords format
    nmax = 100 if debug else -1
    if 'train' in data_type:
        print('\nWriting train tfrecords ...')
        train_path = os.path.join(output_path, 'train/')
        if not tf.gfile.IsDirectory(train_path):
            tf.gfile.MakeDirs(train_path)
        write_tfrecords(train_path, train_it, num_shards, nmax=nmax)

    if 'val' in data_type:
        print('\nWriting val tfrecords ...')
        val_path = os.path.join(output_path, 'val/')
        if not tf.gfile.IsDirectory(val_path):
            tf.gfile.MakeDirs(val_path)
        write_tfrecords(val_path, val_it, num_shards, nmax=nmax)

    if 'test' in data_type:
        print('\nWriting test tfrecords ...')
        test_path = os.path.join(output_path, 'test/')
        if not tf.gfile.IsDirectory(test_path):
            tf.gfile.MakeDirs(test_path)
        write_tfrecords(test_path, test_it, num_shards, nmax=nmax)


def main(_):
    create_tfrecords(FLAGS.data_dir,
                     max_len=FLAGS.max_len,
                     split=FLAGS.data_split,
                     num_shards=FLAGS.num_shards,
                     debug=FLAGS.debug,
                     data_type=FLAGS.data_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
