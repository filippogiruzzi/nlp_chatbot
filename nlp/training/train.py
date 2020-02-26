import os
import glob
import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import json

from nlp.training.input_pipeline import data_input_fn
from nlp.training.estimator import Seq2SeqEstimator


def main():
    parser = argparse.ArgumentParser(description='train the NLP model')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/cornell_movie_data/tfrecords/',
                        help='tf records data directory')
    parser.add_argument('--model-dir', type=str, default='', help='pretrained model directory')
    parser.add_argument('--ckpt', type=str, default='', help='pretrained checkpoint directory')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train, eval or predict')
    parser.add_argument('--model', type=str, default='seq2seq', help='model name')
    parser.add_argument('--batch-size', '-bs', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='train epochs')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--max-len', type=int, default=10, help='max sentence length')
    parser.add_argument('--voc-size', type=int, default=7826, help='vocabulary size')
    parser.add_argument('--loss', type=str, default='regular', help='regular loss or masked loss')
    parser.add_argument('--fake-input', action='store_true', default=False, help='debug with 1 batch training')
    args = parser.parse_args()

    assert args.model in ['seq2seq'], 'Wrong model name'
    assert args.loss in ['regular', 'masked'], 'Wrong loss name'

    tfrecords_train = glob.glob('{}train/*.tfrecord'.format(args.data_dir))
    tfrecords_val = glob.glob('{}val/*.tfrecord'.format(args.data_dir))
    tfrecords_test = glob.glob('{}test/*.tfrecord'.format(args.data_dir))

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not args.model_dir:
        save_dir = '{}models/{}/{}/'.format(args.data_dir, args.model, datetime.now().isoformat())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.model_dir

    params = {
        'model': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.learning_rate,
        'max_len': args.max_len,
        'voc_size': args.voc_size,
        'emb_dim': 256,
        'enc_dim': 128,
        'dec_dim': 128,
        'loss': args.loss
    }

    train_config = tf.estimator.RunConfig(save_summary_steps=10,
                                          save_checkpoints_steps=500,
                                          keep_checkpoint_max=10,
                                          log_step_count_steps=10)

    ws = None
    if args.ckpt:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=args.ckpt, vars_to_warm_start='.*')

    # Create TensorFlow estimator object
    estimator_obj = Seq2SeqEstimator(params)
    estimator = tf.estimator.Estimator(model_fn=estimator_obj.model_fn,
                                       model_dir=save_dir,
                                       config=train_config,
                                       params=params,
                                       warm_start_from=ws)

    mode_keys = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }
    mode = mode_keys[args.mode]

    # Training & Evaluation on Train / Val set
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input_fn = data_input_fn(tfrecords_train,
                                       batch_size=params['batch_size'],
                                       epochs=1,
                                       max_len=params['max_len'],
                                       voc_size=params['voc_size'],
                                       shuffle=True,
                                       fake_input=args.fake_input)
        eval_input_fn = data_input_fn(tfrecords_val,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      max_len=params['max_len'],
                                      voc_size=params['voc_size'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        for epoch_num in range(params['epochs']):
            logger.info("Training for epoch {} ...".format(epoch_num))
            estimator.train(input_fn=train_input_fn)
            logger.info("Evaluation for epoch {} ...".format(epoch_num))
            estimator.evaluate(input_fn=eval_input_fn)

    # Evaluation on Test set
    elif mode == tf.estimator.ModeKeys.EVAL:
        test_input_fn = data_input_fn(tfrecords_val,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      max_len=params['max_len'],
                                      voc_size=params['voc_size'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        logger.info("Evaluation of test set ...")
        estimator.evaluate(input_fn=test_input_fn)

    # Prediction visualization on Test set
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_input_fn = data_input_fn(tfrecords_test,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      max_len=params['max_len'],
                                      voc_size=params['voc_size'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        voc_fp = os.path.join(args.data_dir, 'voc.json')
        with open(voc_fp, 'r') as f:
            voc_data = json.load(f)

        predictions = estimator.predict(input_fn=test_input_fn)
        for n, pred in enumerate(predictions):
            input_sentence = np.int64(pred['input_sentence'])
            input_target = np.int64(pred['input_target'])
            pred = np.int64(np.argmax(pred['answer'], axis=-1))

            input_sentence_words = [voc_data['index2word'][str(x)] for x in input_sentence]
            input_target_words = [voc_data['index2word'][str(x)] for x in input_target]
            pred_words = [voc_data['index2word'][str(x)] for x in pred]

            print("\nInputs:")
            print(input_sentence, input_target)
            print(input_sentence_words, input_target_words)
            print("Prediction:")
            print(pred)
            print(pred_words)


if __name__ == '__main__':
    main()
