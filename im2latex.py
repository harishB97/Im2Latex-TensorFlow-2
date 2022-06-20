# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:03:49 2022

@author: harish
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, BatchNormalization, LSTM, concatenate, Softmax, RNN
import numpy as np

#%% Constants

C = 1 # channels
vocab_size = 502
embedding_dim = 80
ENC_DIM = 256 # Hidden state dimension of encoder RNN
DEC_DIM = 512 # Hidden state dimension of decoder RNN

#%% Define all layers in the model

class EncoderCell(keras.layers.Layer):
    '''
    Splits the convolution output vertically along height (dim == 1) and
    runs RNN on each vertical cross section of conv output
    '''
    def __init__(self, encoder, state_size, output_size, **kwargs):
        self.encoder = encoder
        self.state_size = state_size
        self.output_size = output_size
        super(EncoderCell, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.built = True
 
    def call(self, inputs, states):
        output = self.encoder(inputs)
        return output, states


class AttentionCell(keras.layers.Layer):
    
    def __init__(self, input_embedding_size, decoder_out_shape, state_size, output_size, **kwargs):
        self.input_embedding_size = input_embedding_size
        self.decoder_out_shape = decoder_out_shape
        self.state_size = state_size
        self.output_size = output_size # vocab_size
        self.context_vector = None
        self.i_step_count = 0
        super(AttentionCell, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        
        self.gates = self.add_weight(shape=(self.input_embedding_size[0]+512, 4*512),
                                  initializer=tf.keras.initializers.GlorotUniform(),
                                  trainable=True,
                                  name='gates')  # (80+512, 4*512)
        self.gates_bias = self.add_weight(shape=(1, 4*512),
                                  initializer='zeros',
                                  trainable=True,
                                  name='gates_bias')  # (1, 4*512)
        self.Wa = self.add_weight(shape=(self.decoder_out_shape[1], self.decoder_out_shape[1]),
                                  initializer=tf.keras.initializers.GlorotUniform(),
                                  trainable=True,
                                  name='Wa')  # (512, 512)
        self.Wc = self.add_weight(shape=(self.decoder_out_shape[1]*2, self.decoder_out_shape[1]),
                                  initializer=tf.keras.initializers.GlorotUniform(),
                                  trainable=True,
                                  name='Wc')  # (512 + 512, 512) => (ENC_DIM*2 + DEC_DIM, DEC_DIM)
        self.Ws = self.add_weight(shape=(self.decoder_out_shape[1], self.output_size[0]),
                                  initializer=tf.keras.initializers.GlorotUniform(),
                                  trainable=True,
                                  name='Ws')  # (512, vocab_size)
        self.Bs = self.add_weight(shape=(1, self.output_size[0]),
                                  initializer='zeros',
                                  trainable=True,
                                  name='Bs')  # (1, vocab_size)
        self.built = True

        
    def call(self, inputs, states, constants):
        
        # embed, hs = inputs
        embed = inputs
        hs = constants
        # print('hs', hs)

        """
        source_l => (W//8)*(H//8)
        hs => (None, None, None) => (Batch, source_l, 512)
        embed => (None, 80) => (Batch, 80)
        states => (None, 1536) => (Batch, 1536)
        """
        c_tm1, ht_bar_tm1 = tf.split(axis=-1, num_or_size_splits=2, value=states)
        c_tm1 = tf.squeeze(c_tm1, axis=0) # (1, None, 512) to (None, 512) => (Batch, 512)
        ht_bar_tm1 = tf.squeeze(ht_bar_tm1, axis=0) # (1, None, 512) to (None, 512) => (Batch, 512)
        
        xt = concatenate([embed, ht_bar_tm1], axis=-1) # (None, 512) + (None, 80) => (None, 592) => (Batch, 592)
        gates_out = tf.linalg.matmul(xt, self.gates) + self.gates_bias # (None, 2048) => (Batch, 2048)
        i_t, f_t, o_t, g_t = tf.split(axis=-1, num_or_size_splits=4, value=gates_out) #each (None, 512) => (Batch, 512)

        c_t = tf.math.sigmoid(f_t)*c_tm1 + tf.math.sigmoid(i_t)*tf.tanh(g_t) # (None, 512) => (Batch, 512)
        h_t = tf.math.sigmoid(o_t)*tf.tanh(c_t) # (None, 512) => (Batch, 512)
        
        h_t = tf.expand_dims(h_t, axis=-1) # (None, 512, 1) => (Batch, 512, 1)
        Wa_ht = tf.linalg.matmul(self.Wa, h_t) # (512, 512) * (None, 512, 1)
        score = tf.linalg.matmul(hs, Wa_ht) # (None, None, None) * (None, 512, 1) => (None, None, 1)
        # (Batch, source_l, 512) * (Batch, 512, 1) => (Batch, source_l, 1)
        score = tf.squeeze(score, axis=0) # unexpected dimension in beginning (1, None, None, 1) instead of (None, None, 1)
        score = tf.squeeze(score, axis=-1) # (None, None) => (Batch, source_l)
        # print(2, 'score', score)
        
        at = Softmax(axis=-1)(score) # (None, None) => (Batch, source_l)
        at = tf.expand_dims(at, axis=-2) # (None, 1, None) => (Batch, 1, source_l)
        ct = tf.linalg.matmul(at, hs) # (None, 1, None) * (None, None, None) => (None, 1, None)
        ct = tf.squeeze(ct, axis=0) # unexpected dimension in beginning (1, None, 1, None) instead of (None, 1, None)
        # (Batch, 1, source_l) * (Batch, source_l, 512) => (Batch, 1, 512)
        
        h_t = tf.squeeze(h_t, axis=-1) # (None, 512, 1) to (None, 512) => (Batch, 512)
        h_t = tf.expand_dims(h_t, axis=-2) # (None, 512) to (None, 1, 512) => (Batch, 1, 512)
        ct_h_t = tf.concat([ct, h_t], axis=-1) # (None, 1, None) + (None, 1, 512)
        # (Batch, 1, 512) + (Batch, 1, 512) => (Batch, 1, 1024)
        ct_h_t_Wc = tf.linalg.matmul(ct_h_t, self.Wc) # (None, 1, None) * (1024, 512)
        # (Batch, 1, 1024) * (1024, 512) => (Batch, 1, 512)
        ht_bar = tf.math.tanh(ct_h_t_Wc) # (None, 1, None) => (Batch, 1, 512)
        Ws_ht_bar = tf.linalg.matmul(ht_bar, self.Ws) + self.Bs # (None, 1, None) * (512, vocab) => (None, 1, vocab)
        # (Batch, 1, 512) * (512, vocab) => (Batch, 1, vocab)
        
        output = tf.squeeze(Ws_ht_bar, axis=-2) # (Batch, 1, vocab) to (Batch, vocab)

        # mean normalization on logits (output)
        mean = tf.math.reduce_mean(output, axis=-1)
        mean = tf.expand_dims(mean, axis=-1)
        variance = tf.math.reduce_mean(tf.math.square(output - mean), axis=-1)
        variance = tf.expand_dims(variance, axis=-1)
        output = (output - mean) / tf.math.sqrt(variance)

        h_t = tf.squeeze(h_t, axis=-2) # (Batch, 1, 512) to (Batch, 512)
        ht_bar = tf.squeeze(ht_bar, axis=-2) # (Batch, 1, 512) to (Batch, 512)
        self.i_step_count += 1

        return output, concatenate([c_t, ht_bar], axis=-1) # (Batch, vocab), 


layers = {}


def define_layers():
    layers['conv1'] = Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['maxpool1'] = MaxPool2D(pool_size=[2, 2], strides=[2, 2])
    layers['conv2'] = Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['maxpool2'] = MaxPool2D(pool_size=[2, 2], strides=[2, 2])
    layers['conv3'] = Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['bn1'] = BatchNormalization()
    layers['conv4'] = Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['maxpool3'] = MaxPool2D(pool_size=[1, 2], strides=[1, 2])
    layers['conv5'] = Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['bn2'] = BatchNormalization()
    layers['maxpool4'] = MaxPool2D(pool_size=[2, 1], strides=[2, 1])
    layers['conv6'] = Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu', use_bias=False)
    layers['bn3'] = BatchNormalization()
    
    encoder_fw_cell = EncoderCell(LSTM(ENC_DIM, return_sequences=True), state_size=tf.TensorShape([1]), output_size=tf.TensorShape([None, ENC_DIM]))
    encoder_bw_cell = EncoderCell(LSTM(ENC_DIM, return_sequences=True, go_backwards=True), state_size=tf.TensorShape([1]), output_size=tf.TensorShape([None, ENC_DIM]))
    
    layers['encoder_fw'] = RNN(encoder_fw_cell, return_sequences=True)
    layers['encoder_bw'] = RNN(encoder_bw_cell, return_sequences=True)
    
    

    layers['embedding'] = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                                   embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=1.0/np.sqrt(vocab_size)))
    
    layers['attention_cell'] = AttentionCell(input_embedding_size=tf.TensorShape([embedding_dim]), 
                                decoder_out_shape=tf.TensorShape([None, DEC_DIM]),
                                state_size=tf.TensorShape([DEC_DIM*2]),# tf.TensorShape([None, None, None])], #tf.TensorShape([DEC_DIM*3]), 
                                output_size=tf.TensorShape([vocab_size]))
    
    layers['attention_layer'] = RNN(layers['attention_cell'], return_sequences=True, return_state=True)

#%% Build the model with the layers
def build_model(image, latex_seq, encoder_hid_st_input=None, decoder_init_state=None):
    # encoder
    img = image-128
    img = img/128

    x = layers['conv1'](img)
    x = layers['maxpool1'](x)
    # x -> (H/2, W/2, 64)

    x = layers['conv2'](x)
    x = layers['maxpool2'](x)
    # x -> (H/4, W/4, 128)

    x = layers['conv3'](x)
    x = layers['bn1'](x)
    # x -> (H/4, W/4, 256)

    x = layers['conv4'](x)
    x = layers['maxpool3'](x)
    # x -> (H/4, W/8, 256)

    x = layers['conv5'](x)
    x = layers['bn2'](x)
    x = layers['maxpool4'](x)
    # x -> (H/8, W/8, 512)

    x = layers['conv6'](x)
    x = layers['bn3'](x)
    # x -> (H/8, W/8, 512)
    
    encoder_fw_hid_st = layers['encoder_fw'](x)
    encoder_fw_hid_st = tf.reshape(encoder_fw_hid_st,[tf.shape(encoder_fw_hid_st)[0],-1,tf.shape(encoder_fw_hid_st)[-1]])
    
    encoder_bw_hid_st = layers['encoder_bw'](x)
    encoder_bw_hid_st = tf.reshape(encoder_bw_hid_st,[tf.shape(encoder_bw_hid_st)[0],-1,tf.shape(encoder_bw_hid_st)[-1]])

    encoder_hid_st = concatenate([encoder_fw_hid_st, encoder_bw_hid_st], axis=-1)
    
    # decoder
    if encoder_hid_st_input is None: # training
        latex_emb = layers['embedding'](latex_seq)        
        logits = layers['attention_layer'](inputs=latex_emb, constants=encoder_hid_st)
        return keras.Model(inputs=[image, latex_seq], outputs=logits)
    else: # inference
        latex_emb = layers['embedding'](latex_seq)
        logits, decoder_state = layers['attention_layer'](inputs=latex_emb,
                                                          initial_state=decoder_init_state,
                                                          constants=encoder_hid_st_input)
        return (keras.Model(inputs=[image], outputs=encoder_hid_st), 
                keras.Model(inputs=[latex_seq, encoder_hid_st_input, decoder_init_state],
                            outputs=[logits, decoder_state]))

#%% Loss & Optimizer
optimizer = tf.keras.optimizers.SGD(clipnorm=5)
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

def loss_func(y_true, logits):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                               reduction=tf.keras.losses.Reduction.NONE)
    loss = ce(y_true, logits)
    mask = tf.cast((y_true != 0), tf.float32)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


#%% Build training and inference models
def build_training_model(saved_model=None):
    define_layers()
    im2latex_training_model = build_model(Input(shape=(None, None, C)), 
                                    Input(shape=tf.TensorShape([None])),
                                    None)
    if saved_model is not None:
        im2latex_training_model.load_weights(saved_model)
        print("Model loaded from", saved_model)
        
    im2latex_training_model.compile(optimizer=optimizer, 
                                    loss=loss_func, 
                                    metrics=['accuracy', 'crossentropy'])
    
    return im2latex_training_model


def build_inference_model(saved_model):
    build_training_model(saved_model)
    im2latex_inference_encoder, im2latex_inference_decoder = build_model(Input(shape=(None, None, C)), 
                                                                             Input(shape=tf.TensorShape([None])), 
                                                                             Input(shape=(None, ENC_DIM*2)),
                                                                             Input(shape=(DEC_DIM*2)))
    
    return im2latex_inference_encoder, im2latex_inference_decoder
