#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:53:49 2024

@author: johnmiller

Transformer decoder model for word character sequences.
Separate from training and model application.
"""
import attr

# import tensorflow as tf
# tf.config.list_physical_devices("GPU")
import keras, keras_nlp

import lexbor.config as cfg

@attr.s
class TransformerDecoderWM():
    """
    Transformer model for word character sequences.


    Parameters
    ----------
    vocab_size: int
        size of vocab.
    settings: data from config
        Default settings for various modules and methods.

    """

    vocab_size = attr.ib(None)
    settings = attr.ib(default=cfg.NeuralSettings())
    model = attr.ib(init=False)
    softmax_model = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__init__()

        inputs = keras.Input(
            shape=(None,),
            dtype="int32",
            name="token_ids",
        )

        embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=self.vocab_size,
            sequence_length=self.settings.token_maxlen,
            embedding_dim=self.settings.embedding_len,
            mask_zero=False
        )
        outputs = embedding_layer(inputs)

        outputs = keras.layers.LayerNormalization(  # Is this layer required?
            epsilon=self.settings.norm_epsilon)(outputs)
        outputs = keras.layers.Dropout(
            rate=self.settings.embedding_dropout)(outputs)

        for _ in range(self.settings.transformer_layers):
            outputs = keras_nlp.layers.TransformerDecoder(
                num_heads=self.settings.num_heads,
                intermediate_dim=self.settings.model_dim,
                dropout=self.settings.transformer_dropout,
                layer_norm_epsilon=self.settings.norm_epsilon
                )(outputs)


        logits = keras.layers.Dense(self.vocab_size)(outputs)

        self.model = keras.Model(inputs=inputs, outputs=logits,
                                 name="Transformer_Decoder_Word_Model")

        outputs = keras.layers.Softmax()(logits)

        self.softmax_model = keras.Model(inputs=inputs, outputs = outputs,
                                name="Transformer_Decoder_Word_Model_Softmax")

