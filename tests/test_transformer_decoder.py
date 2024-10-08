#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:25:40 2024

@author: johnmiller

Test transformer_decoder.py

All test functions begin with 'test_'

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_transformer_decoder.py -—cov=lexbor.transformer_decoder

Or to save as html report:
$ pytest tests/test_transformer_decoder.py -—cov=lexbor.transformer_decoder --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_transformer_decoder.py --cov=lexbor.transformer_decoder --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_transformer_decoder.py::test_prep_data --cov=lexbor.transformer_decoder --cov-report term-missing
$ pytest tests/test_transformer_decoder.py::test_train --cov=lexbor.transformer_decoder --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

Add -s to capture output.

"""
# import statistics
import lexbor.neural_data as nd
import lexbor.transformer_decoder as td
import lexbor.config as cfg
from test_neural_data import get_data_for_language
import pytest


# Import for development of model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.config.list_physical_devices("GPU")
import keras, keras_nlp

from lexbor.constants import DataFmtConst

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================d

def test_devel():
    print("test - devel transformer decoder")

    train_ds, test_ds = get_data_for_language(
        language='English', form='Tokens')
    neuraldata = nd.NeuralData(train_ds, test_ds, val_split=0.2)
    train, test = neuraldata.get_token_datasets()
    fit, val = neuraldata.get_fit_val_datasets(train)

    for x_train, y_train in train.take(1): break
    print(x_train.shape)
    print(y_train.shape)
    # for x_train, y_train in train.unbatch().take(1): break
    # print(x_train, y_train)

    settings = cfg.NeuralSettings()

    token_id_input = keras.Input(
        shape=(None,), dtype="int32", name="token_ids")

    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=neuraldata.vocab.size,
        sequence_length=settings.token_maxlen,
        embedding_dim=settings.embedding_len,
        mask_zero=True
    )
    outputs = embedding_layer(token_id_input)
    print(outputs.shape)

    outputs = keras.layers.LayerNormalization(
        epsilon=settings.norm_epsilon)(outputs)
    outputs = keras.layers.Dropout(
        rate=settings.embedding_dropout)(outputs)
    print(outputs.shape)

    outputs = keras_nlp.layers.TransformerDecoder(
        num_heads=settings.num_heads,
        intermediate_dim=settings.model_dim,
        dropout=settings.transformer_dropout,
        layer_norm_epsilon=settings.norm_epsilon
        )(outputs)
    print(outputs.shape)
    # print(outputs[:2][:4][:10])

    outputs = keras.layers.Dense(neuraldata.vocab.size)(outputs)
    print(outputs.shape)
    # print(outputs[:2])

    transformer = keras.Model(inputs=token_id_input, outputs=outputs)
    transformer.summary()


def test_instantiation():
    print("test - instantiation")

    # Use typical vocabulary size to test out instantiation.
    train_ds, test_ds = get_data_for_language(language='English', form='Tokens')
    neuraldata = nd.NeuralData(train_ds, test_ds, val_split=0.2)

    # train, test = neuraldata.get_token_datasets()
    # fit, val = neuraldata.get_fit_val_datasets(train)

    neural_dec = td.NeuralDecoder(neuraldata.vocab.size, cfg.NeuralSettings())
    neural_dec.make_decoder_model()

    neural_dec.transformer.summary()


def test_schedule():
    print("test - schedule for use with Adam optmizer")

    # settings = cfg.NeuralSettings()
    learning_rate = td.TRSchedule(model_dim=32, warmup_steps=10)
    assert learning_rate.model_dim == 32.0
    assert learning_rate.warmup_steps == 10
    assert pytest.approx(learning_rate.learning_rate, abs=0.00002) == 0.17678
    # Could use relative tolerance as well with, e.g., rel=0.001
    for step in [0, 1, 2, 10, 11, 12, 50, 100, 1000, 10000]:
        print(step, learning_rate(step))
    """ Expected:
        step : learning_rate
        0: 0
        1: 0.00559
        2: 0.01118
        10: 0.05590
        11: 0.53300
        12: 0.51031
        50: 0.02500
        100: 0.017678
        1000: 0.0055902
        10000: 0.0017678
    """


def test_train():
    print("test - train decoder")

    # Use typical vocabulary size to test out instantiation.
    train_ds, test_ds = get_data_for_language(language='English', form='Tokens')
    neuraldata = nd.NeuralData(train_ds, test_ds, val_split=0.2)

    train, test = neuraldata.get_token_datasets()
    fit, val = neuraldata.get_fit_val_datasets(train)
    # print("fit, val lengths:",
    #       len(list(fit.unbatch().as_numpy_iterator())),
    #       len(list(val.unbatch().as_numpy_iterator())))

    neural_dec = td.NeuralDecoder(neuraldata.vocab.size, cfg.NeuralSettings())
    neural_dec.make_decoder_model()
    fit = fit.shuffle(10000).repeat(10)
    val = val.shuffle(10000).repeat(10)
    neural_dec.train(fit, val=val, epochs=2)

    test = test.shuffle(10000).repeat(10)
    measures = neural_dec.evaluate_test(test)
    print(f"Test dataset: loss={measures[0]:.4f}, accuracy={measures[1]:.4f}")

def test_train_gen():
    print("test - train decoder - with generators")

    # Use typical vocabulary size to test out instantiation.
    train_ds, test_ds = get_data_for_language(
        language='English', form='Tokens', test_split=0.15)
    neuraldata = nd.NeuralDataGen(train_ds, test_ds, val_split=0.15)

    test = neuraldata.test
    fit = neuraldata.fit
    val = neuraldata.val

    neural_dec = td.NeuralDecoder(neuraldata.vocab.size,
                                  settings=cfg.NeuralSettings())
    neural_dec.make_decoder_model()
    print(neural_dec.model.summary())

    neural_dec.train(fit, val=val, epochs=50)

    measures = neural_dec.evaluate_test(test)
    print(f"Test dataset: loss={measures[0]:.4f}, accuracy={measures[1]:.4f}")


def test_calculate_entropies():
    print("test - train decoder - calculate entropies")

    train_ds, test_ds = get_data_for_language(
        language='English', form='Tokens', test_split=0.15)
    neuraldata = nd.NeuralDataGen(train_ds, test_ds, val_split=0.15)

    # generators for testing.
    test = neuraldata.test
    fit = neuraldata.fit
    val = neuraldata.val

    neural_dec = td.NeuralDecoder(neuraldata.vocab.size,
                                  settings=cfg.NeuralSettings())
    neural_dec.make_decoder_model()
    print(neural_dec.model.summary())

    neural_dec.train(fit, val=val)

    measures = neural_dec.evaluate_test(test)
    print(f"Test dataset: loss={measures[0]:.4f}, accuracy={measures[1]:.4f}")

    # Use original wordlists with segmented IPA for entropies.
    pkg = neuraldata.pkg  # Get PackageSequences - this includes Vocab.
    entropies = neural_dec.calculate_entropies(data=test_ds, pkg=pkg)

    print("Entropies", entropies[:10])
    print("Test_ds", test_ds[:10])

    import statistics
    print("Entropies:", "Mean =", statistics.mean(entropies),
          "Stdev =", statistics.stdev(entropies))
