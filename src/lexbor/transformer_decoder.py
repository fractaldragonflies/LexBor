#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:45:29 2024

@author: johnmiller
"""
"""
Transformer decoder model.
Provide functions to train model and calculate word entropies.
"""

# import numpy as np
import math
import attr
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.config.list_physical_devices("GPU")
import keras, keras_nlp
from tensorflow import clip_by_value

# import matplotlib.pyplot as plt
# import seaborn as sns

import lexbor.config as cfg
import lexbor.neural_data as nd
import lexbor.transformer_decoder_wm as tdwm
import lexbor.loss_metric_funcs as lmf
from lexbor.constants import DataFmtConst, detect_input_fmt

EPSILON = 1e-7  # Used with prediction for clipping.

# metric_object = tf.keras.metrics.SparseCategoricalCrossentropy(
#     from_logits=True)

@attr.s
class NeuralDecoder():
    """
    Provides neural model, training, and test for word character sequences.


    Parameters
    ----------
    vocab_size: int
        size of vocab.
    settings: data from Config
        Default settings for various modules and methods.

    """

    vocab_size = attr.ib(None)
    settings = attr.ib(default=cfg.NeuralSettings())
    model = attr.ib(init=False)
    softmax_model = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__init__()

    def train(self, fit=None, val=None, epochs=None, learning_rate=None):
        """
        Train neural network using train and val datasets.

        Parameters
        ----------
        fit : KerasBatchGenerator
            Dataset of segmented IPA from wordlist. The default is None.
        val : KerasBatchGenerator, optional
            Dataset of segmented IPA from wordlist. The default is None.
        epochs : int, optional
            Number epochs training. Overrides config. The default is None.
        learning_rate: float, optional
            Learning rate other than 1.0 to use in learning schedule.

        Returns
        -------
        Tensorflow history.history.

        """

        self.make_decoder_model()

        lr_schedule =  TRSchedule(self.settings.model_dim)
        optimizer = tf.keras.optimizers.Adam(
            lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model.compile(loss=self.masked_smoothed_loss,
                           optimizer=optimizer,
                           metrics=[self.masked_accuracy,
                                    self.masked_crossentropy])

        epochs = epochs if epochs is not None else self.settings.epochs

# TODO test for whether fit, val, test, train are generator or dataset.
# Adjust model fit to take into account.
        if isinstance(fit, tf.data.Dataset):  #nd.NeuralBatchDataset):
            if val is None:
                history = self.model.fit(
                    fit.shuffle(len(fit)).batch(fit.batch_size)
                    .prefetch(tf.data.AUTOTUNE), epochs=epochs)
            else:
                history = self.model.fit(
                    fit.shuffle(len(fit)).batch(fit.batch_size)
                    .prefetch(tf.data.AUTOTUNE), epochs=epochs,
                    validation_data=val.shuffle(len(val)).batch(val.batch_size)
                    .prefetch(tf.data.AUTOTUNE))

        elif isinstance(fit, nd.KerasBatchGenerator):
            fit_steps = math.ceil(len(fit)/fit.batch_size)
            if val is None:
                history = self.model.fit(fit.generate(),
                                         steps_per_epoch=fit_steps,
                                         epochs=epochs)
            else:
                val_steps = math.ceil(len(val)/val.batch_size)
                history = self.model.fit(fit.generate(),
                                         steps_per_epoch=fit_steps,
                                         epochs=epochs,
                                         validation_data=val.generate(),
                                         validation_steps=val_steps)
        else:
            print("Type of fit dataset:", type(fit))
            raise TypeError(
                "Dataset must be KerasBatchGenerator or NeuralBatchDataset")

        return history.history

    def show_test(self, test=None):
        if not test: return

        results = self.evaluate_test(test)
        if results:
            print(f"Test dataset: loss={results[0]:.4f}, "
                  + f"accuracy={results[1]:.4f}, "
                  + f"crossentropy={results[2]:.4f}.")

    def evaluate_test(self, test=None):
        # Evaluate using generator - use evaluate directly.
        if not test: return

        if isinstance(test, tf.data.Dataset):  # NeuralBatchDataset):
            results = self.model.evaluate(test.batch(test.batch_size)
                                          .prefetch(tf.data.AUTOTUNE))
        else:
            eval_steps = math.ceil(len(test)/test.batch_size)
            results = self.model.evaluate(test.generate(), steps=eval_steps)

        return results


    def make_decoder_model(self):
        """
        Construct decoder transformer model.

        Returns
        -------
        None.

        """

        trans_dec = tdwm.TransformerDecoderWM(self.vocab_size, self.settings)
        self.model = trans_dec.model
        self.softmax_model = trans_dec.softmax_model


    # Borrowed from "Imlementing Transformer decoder for text generation
    # in Keras and Tensorflow" by Derrick Mwiti. Last accessed Sep 19, 2024.
    # URL: https://www.machinelearningnuggets.com/transformer-decoder/
    @staticmethod
    def masked_loss(label, pred):
        # masked sparse categorical crossentropy.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none")
        loss = loss_object(label, pred)

        mask = tf.math.not_equal(label, 0)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss
    masked_loss.__full_name__ = "masked_sparse_categorical_crossentropy_loss"

    @staticmethod
    def masked_smoothed_loss(label, pred):
        return lmf.sparseCategoricalCELoss(label, pred,
                                           zero_masking=True,
                                           from_logits=True,
                                           label_smoothing=0.2,
                                           smoothing_policy="both")
    masked_smoothed_loss.name = "masked_smoothed_sparse_CE_loss"

    @staticmethod
    def masked_accuracy(label, pred):
        # masked categorical accuracy.
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = tf.math.equal(label, pred)

        mask = tf.math.not_equal(label, 0)
        # match = match & mask
        match &= mask
        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match) / tf.reduce_sum(mask)

    @staticmethod
    def masked_crossentropy(y_true, y_pred):
        # Masked sparse categorical crossentropy.
        metric_function = tf.keras.metrics.sparse_categorical_crossentropy
        metric = metric_function(y_true, y_pred, from_logits=True)

        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, dtype=metric.dtype)

        metric *= mask
        return tf.reduce_mean(tf.reduce_sum(metric) / tf.reduce_sum(mask))
    masked_crossentropy.__full_name__ = "masked_sparse_categorical_crossentropy_metric"


    def calculate_entropies(self, data, pkg:nd.PackageSequences):
        """
        Compute the crossentropy from a list of words/tokens.
        data: [[str, [str], int]] or [[str]]
            Where [str] correspond to segmented IPA.
        pkg: PackageSequences
            Must be the same as used to tokenize and package training, test data.
        """

        assert data is not None and len(data) > 0
        # Convert IPA segmented data to lists of token_ids.
        fmt = detect_input_fmt(data)
        if fmt == DataFmtConst.UNK:
            print("Error: Unknown data format for calculating entropies.")
            return None
        if fmt == DataFmtConst.STD:
            data = [row[1] for row in data]
        # Data is list of lists of IPA segments [[IPA segments]].

        token_ids_x, token_ids_y = pkg.process_segments(data)

        # Get the probabilities for all str segment.
        y_probs = self.softmax_model.predict(token_ids_x)

        # Compute crossentropies

        entropies = []
        for y_ids_probs, y_ids in zip(y_probs, token_ids_y):
            # Prevent overflow/underflow with clipping.
            y_ids_probs_ = clip_by_value(y_ids_probs,  EPSILON, 1-EPSILON)
            y_ids_lns = [
                math.log(y_ids_probs_[i, y_ids[i]])
                for i in range(len(y_ids)) if y_ids[i] != 0]
                # for i in range(min(maxlen, len(y_ids)))

            # entropy = -sum(y_ids_lns) / math.pow(len(y_ids_lns), self.settings.len_power)
            entropy = -sum(y_ids_lns) / len(y_ids_lns)
            entropies.append(entropy)

        assert len(data) == len(entropies)
        return entropies


@attr.s
class TRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Training schedule for use with Adam optimizer as in
        "Attention is all you need" paper by Vaswani et al 2017+

        Parameters
        ----------
        model_dim: int32,
            Transformer decoder layer model dimension.
        warmup_steps: int32,
            Number of warmup steps for model training.
        learning_rate: float32, 1.0
            Learning rate initial value allows other than 1 before
            calculation as given in Vaswani paper.
        settings: dict
            Dictionary of default settings for this application.
    """

    model_dim=attr.ib(default=None)
    warmup_steps=attr.ib(default=None)
    learning_rate=attr.ib(default=None)
    settings=attr.ib(default=cfg.NeuralSettings())

    def __attrs_post_init__(self):
        super(TRSchedule, self).__init__()
        self.model_dim = (self.model_dim
                          if self.model_dim is not None
                          else self.settings.model_dim)
        self.model_dim = tf.cast(self.model_dim, dtype=tf.float32)
        self.warmup_steps = (self.warmup_steps
                             if self.warmup_steps is not None
                             else self.settings.warmup_steps)
        self.warmup = self.warmup_steps ** -1.5  # Remains constant.
        self.learning_rate = (self.learning_rate
                              if self.learning_rate is not None
                              else self.settings.learning_rate)
        self.learning_rate *= tf.math.rsqrt(self.model_dim)  # Remains constant.

    def __call__(self, step):
        # Calculations depend on step and previous factors.
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup
        return self.learning_rate * tf.math.minimum(arg1, arg2)
