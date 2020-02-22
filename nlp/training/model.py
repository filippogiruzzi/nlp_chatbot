from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, TimeDistributed, Dense


class Seq2Seq(Model):
    def __init__(self, params=None, is_training=False):
        super(Seq2Seq, self).__init__()
        self.is_training = is_training

        self.max_len = params['max_len']
        self.voc_size = params['voc_size']
        self.emb_dim = params['emb_dim']
        self.dec_dim = params['dec_dim']

        # Encoder-Decoder structure
        self.embedding = Embedding(input_dim=self.voc_size,
                                   output_dim=self.emb_dim,
                                   input_length=self.max_len,
                                   name='embedding')
        self.encoder = CuDNNLSTM(self.emb_dim, return_state=True, name='encoder_rnn')
        self.decoder = CuDNNLSTM(self.dec_dim, return_state=True, return_sequences=True, name='decoder_rnn')
        self.fc = TimeDistributed(Dense(self.voc_size, activation='linear'), name='fc')

    def call(self, inputs, training=None, mask=None):
        enc_input = inputs['input_sentence']
        dec_input = inputs['input_target']

        with tf.name_scope('encoder'):
            enc_emb = self.embedding(enc_input)
            enc_out, h, c = self.encoder(enc_emb)
        with tf.name_scope('decoder'):
            dec_emb = self.embedding(dec_input)
            dec_out, _, _ = self.decoder(dec_emb, initial_state=[h, c])

        outputs = self.fc(dec_out)
        return outputs
