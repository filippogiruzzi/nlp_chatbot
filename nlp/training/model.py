from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Embedding, LSTM, TimeDistributed, Dense, ReLU, Dropout


class Seq2Seq(Model):
    def __init__(self, params=None, is_training=False):
        super(Seq2Seq, self).__init__()
        self.is_training = is_training

        self.max_len = params['max_len']
        self.voc_size = params['voc_size']
        self.emb_dim = params['emb_dim']
        self.enc_dim = params['enc_dim']
        self.dec_dim = params['dec_dim']

        # Encoder-Decoder structure
        self.embedding = Embedding(input_dim=self.voc_size,
                                   output_dim=self.emb_dim,
                                   input_length=self.max_len,
                                   name='embedding')
        self.encoder = LSTM(self.enc_dim, return_state=True, name='encoder_rnn')
        self.decoder = LSTM(self.dec_dim, return_state=True, return_sequences=True, name='decoder_rnn')
        self.dec_relu = ReLU(name='decoder_relu')
        self.dec_drop = Dropout(0.1, name='decoder_drop')
        self.fc = TimeDistributed(Dense(self.voc_size, activation=None), name='fc')

    def call(self, inputs, training=None, mask=None):
        enc_input = inputs['input_sentence']
        dec_input = inputs['input_target']

        with tf.name_scope('encoder'):
            enc_emb = self.embedding(enc_input)
            enc_out, h, c = self.encoder(enc_emb)
        with tf.name_scope('decoder'):
            dec_emb = self.embedding(dec_input)
            dec_emb = self.dec_drop(dec_emb)
            dec_emb = self.dec_relu(dec_emb)
            dec_out, _, _ = self.decoder(dec_emb, initial_state=[h, c])

        outputs = self.fc(dec_out)
        return outputs


class Attention(Model):
    def __init__(self, params=None, is_training=False):
        super(Attention, self).__init__()
        self.is_training = is_training

        self.max_len = params['max_len']
        self.emb_dim = params['emb_dim']

        self.attn_weights = TimeDistributed(Dense(self.emb_dim, activation='softmax'), name='attn_weights')
        self.attn_out = TimeDistributed(Dense(self.emb_dim, activation=None), name='attn_out')

    def call(self, inputs, training=None, mask=None):
        dec_emb, enc_out, h = inputs

        hidden = tf.tile(tf.expand_dims(h, axis=1), multiples=[1, self.max_len, 1])
        x = tf.concat([dec_emb, hidden], axis=-1)
        attn_weights = self.attn_weights(x)
        x = tf.multiply(attn_weights, enc_out)
        attn_out = self.attn_out(x)
        return attn_out


class Seq2SeqAttn(Model):
    def __init__(self, params=None, is_training=False):
        super(Seq2SeqAttn, self).__init__()
        self.is_training = is_training

        self.max_len = params['max_len']
        self.voc_size = params['voc_size']
        self.emb_dim = params['emb_dim']
        self.enc_dim = params['enc_dim']
        self.dec_dim = params['dec_dim']

        # Encoder-Decoder structure
        self.embedding = Embedding(input_dim=self.voc_size,
                                   output_dim=self.emb_dim,
                                   input_length=self.max_len,
                                   name='embedding')
        self.encoder = LSTM(self.enc_dim * 2, return_state=True, return_sequences=True, name='encoder_rnn')
        self.decoder = LSTM(self.dec_dim * 2, return_state=True, return_sequences=True, name='decoder_rnn')
        self.attn = Attention(params=params, is_training=is_training)
        self.dec_relu = ReLU(name='decoder_relu')
        self.dec_drop = Dropout(0.1, name='decoder_drop')
        self.fc = TimeDistributed(Dense(self.voc_size, activation=None), name='fc')

    def call(self, inputs, training=None, mask=None):
        enc_input = inputs['input_sentence']
        dec_input = inputs['input_target']

        with tf.name_scope('encoder'):
            enc_emb = self.embedding(enc_input)
            enc_out, h, c = self.encoder(enc_emb)
        with tf.name_scope('decoder'):
            dec_emb = self.embedding(dec_input)
            dec_emb = self.dec_drop(dec_emb)
            dec_emb = self.attn(inputs=[dec_emb, enc_out, h])
            dec_emb = self.dec_relu(dec_emb)
            dec_out, _, _ = self.decoder(dec_emb, initial_state=[h, c])

        outputs = self.fc(dec_out)
        return outputs
