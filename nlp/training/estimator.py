import tensorflow as tf

from nlp.training.model import Seq2Seq, Seq2SeqAttn


class Seq2SeqEstimator(object):
    def __init__(self, params):
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        if params['model'] == 'seq2seq':
            self.model = Seq2Seq(params=params, is_training=training)
        elif params['model'] == 'attn':
            self.model = Seq2SeqAttn(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    @staticmethod
    def loss_fn(labels, predictions, params):
        pred = predictions['answer']
        one_hot_label = labels['one_hot_label']

        loss = None
        if params['loss'] == 'regular':
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=pred)
            loss = tf.reduce_sum(losses) / params['batch_size']
        elif params['loss'] == 'masked':
            label = labels['label']
            loss_mask = label > 0
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=pred)
            masked_losses = tf.where(loss_mask, losses, tf.zeros_like(losses))
            loss = tf.reduce_sum(masked_losses) / params['batch_size']

        tf.summary.scalar('loss', tensor=loss)
        return loss

    @staticmethod
    def accuracy_fn(labels, predictions, params):
        pred = predictions['answer']
        label = labels['label']

        label = tf.expand_dims(label, axis=-1)
        mask = label > 0
        pred_val = tf.expand_dims(tf.argmax(tf.nn.softmax(pred), axis=-1), axis=-1)

        masked_pred = tf.squeeze(tf.where(mask, pred_val, tf.zeros_like(pred_val)), axis=-1)
        total = params['batch_size'] * params['max_len']
        trues = total - tf.reduce_sum(tf.count_nonzero(tf.subtract(masked_pred, tf.squeeze(label, axis=-1)), axis=-1))
        accuracy = trues / total
        return accuracy

    def model_fn(self, features, labels, mode, params):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        preds = self._output_network(features, params, training=training)

        # Training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = None
            if params['optimizer'] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=params['lr'])
            elif params['optimizer'] == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

            predictions = {'answer': preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(loss, optimizer, global_step=tf.train.get_global_step())

            acc = self.accuracy_fn(labels, predictions, params)
            tf.summary.scalar('acc', tensor=acc, family='accuracy')
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Evaluation op
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {'answer': preds}
            # pred_val = tf.one_hot(tf.argmax(tf.nn.softmax(predictions['answer']), -1), params['voc_size'])
            # acc = tf.metrics.accuracy(labels=labels['one_hot_label'], predictions=pred_val)
            # metrics = {
            #     'accuracy/accuracy/acc': acc
            # }

            loss = self.loss_fn(labels, predictions, params)
            # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

        # Prediction op
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'input_sentence': features['input_sentence'],
                'input_target': features['input_target'],
                'answer': tf.nn.softmax(preds)
            }
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
