import os
from absl import flags, app

import numpy as np
import tensorflow as tf
import json


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/cornell_movie_data/tfrecords/',
                    'path to data directory')
flags.DEFINE_string('exported_model',
                    '/home/filippo/datasets/cornell_movie_data/tfrecords/models/seq2seq/inference/exported/',
                    'path to pretrained TensorFlow exported model')
flags.DEFINE_integer('max_len',
                     10,
                     'max sentence length')
flags.DEFINE_integer('voc_size',
                     7826,
                     'vocabulary size')
FLAGS = flags.FLAGS


def print_answer(answer_str):
    out_str = ''
    for s in answer_str:
        if s == 'EOS':
            break
        out_str += '{} '.format(s)
    print(out_str[:-1])


def main(_):
    np.random.seed(0)

    # TensorFlow inputs
    input_sentence_ph = tf.placeholder(shape=[FLAGS.max_len], dtype=tf.float32)
    input_sentence_op = tf.expand_dims(input_sentence_ph, axis=0)
    input_target_ph = tf.placeholder(shape=[FLAGS.max_len], dtype=tf.float32)
    input_target_op = tf.expand_dims(input_target_ph, axis=0)

    # TensorFlow exported model
    nlp_predictor = tf.contrib.predictor.from_saved_model(export_dir=FLAGS.exported_model)
    init = tf.initializers.global_variables()

    # Load vocabulary data
    voc_fp = os.path.join(FLAGS.data_dir, 'voc.json')
    with open(voc_fp, 'r') as f:
        voc_data = json.load(f)

    with tf.Session() as sess:
        sess.run(init)
        while True:
            print("\n>")
            str_input = input()
            str_input = str_input.split(' ')
            if len(str_input) > FLAGS.max_len - 1:
                print("Too many words, try again !!!")
                continue

            try:
                input_sentence = [voc_data['word2index'][x] for x in str_input]
            except KeyError as e:
                print("Word {} unknown, try again !!!".format(e))
                continue
            input_sentence.append(2)
            while len(input_sentence) < FLAGS.max_len:
                input_sentence.append(0)

            input_sentence = np.float32(input_sentence)
            input_target_np = np.ones_like(input_sentence, dtype=np.float32)

            input_sentence = sess.run(input_sentence_op, feed_dict={input_sentence_ph: input_sentence})
            pt = 1
            while pt < len(input_target_np):
                input_target = sess.run(input_target_op, feed_dict={input_target_ph: input_target_np})

                answer_prob = nlp_predictor({'input_sentence': input_sentence,
                                             'input_target': input_target})['answer'][0]
                answer = np.int64(np.argmax(answer_prob, axis=-1))

                input_target_np[pt] = np.float32(answer[pt - 1])
                pt += 1

            answer_str = [voc_data['index2word'][str(x)] for x in answer]
            print_answer(answer_str)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
