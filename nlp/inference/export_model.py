import argparse
import logging
import sys
import numpy as np
import tensorflow as tf

from nlp.training.estimator import Seq2SeqEstimator


def main():
    parser = argparse.ArgumentParser(description='export trained TensorFlow model for inference')
    parser.add_argument('--model-dir', '-md', type=str, default='', help='pretrained model directory')
    parser.add_argument('--ckpt', '-c', type=str, default='', help='pretrained checkpoint directory')
    parser.add_argument('--model', '-mo', type=str, default='seq2seq', help='model name')
    parser.add_argument('--max-len', '-ml', type=int, default=10, help='max sentence length')
    parser.add_argument('--voc-size', '-vs', type=int, default=7826, help='vocabulary size')
    args = parser.parse_args()

    assert args.model in ['seq2seq', 'attn'], 'Wrong model name'

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    save_dir = args.model_dir

    params = {
        'model': args.model,
        'max_len': args.max_len,
        'voc_size': args.voc_size,
        'emb_dim': 256,
        'enc_dim': 128,
        'dec_dim': 128,
    }

    train_config = tf.estimator.RunConfig(save_summary_steps=10,
                                          save_checkpoints_steps=500,
                                          keep_checkpoint_max=10,
                                          log_step_count_steps=10)

    estimator_obj = Seq2SeqEstimator(params)
    estimator = tf.estimator.Estimator(model_fn=estimator_obj.model_fn,
                                       model_dir=save_dir,
                                       config=train_config,
                                       params=params)

    feature_spec = {
        'input_sentence': tf.placeholder(shape=[1, args.max_len], dtype=tf.float32),
        'input_target': tf.placeholder(shape=[1, args.max_len], dtype=tf.float32)
    }

    logger.info('Exporting TensorFlow trained model ...')
    raw_serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec, default_batch_size=1)
    estimator.export_savedmodel(save_dir, raw_serving_fn, strip_default_attrs=True)


if __name__ == '__main__':
    main()
