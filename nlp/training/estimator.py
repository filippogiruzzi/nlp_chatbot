import tensorflow as tf

from nlp.training.model import Seq2Seq


class Seq2SeqEstimator(object):
    def __init__(self, params):
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        if params['model'] == 'seq2seq':
            self.model = Seq2Seq(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    @staticmethod
    def loss_fn(labels, predictions, params):
        pred = predictions['answer']
        label = labels['label']

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred)
        loss = tf.reduce_sum(losses) / params['batch_size']
        tf.summary.scalar('loss', tensor=loss)
        return loss

    def model_fn(self, features, labels, mode, params):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        preds = self._output_network(features, params, training=training)

        # Training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=params['lr'])

            predictions = {'answer': preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(loss, optimizer, global_step=tf.train.get_global_step())

            pred_val = tf.one_hot(tf.argmax(tf.nn.softmax(predictions['answer']), -1), params['voc_size'])
            acc = tf.metrics.accuracy(labels=labels['label'], predictions=pred_val)
            tf.summary.scalar('acc', tensor=acc[1], family='accuracy')
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Evaluation op
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {'answer': preds}
            pred_val = tf.one_hot(tf.argmax(tf.nn.softmax(predictions['answer']), -1), params['voc_size'])
            metrics = {
                'accuracy/accuracy/acc': tf.metrics.accuracy(labels=labels['label'], predictions=pred_val)
            }

            loss = self.loss_fn(labels, predictions, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

        # Prediction op
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'input_sentence': features['input_sentence'],
                'input_target': features['input_target'],
                'answer': tf.nn.softmax(preds)
            }
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
