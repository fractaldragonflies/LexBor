#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 3, 2024

@author: johnmiller

Test loss_metric_funcs.py

All test functions begin with 'test_'
Install pytest and pytest-cov with pip
Save test files to ./tests

Run test file:
$ pytest tests/test_loss_metric_funcs.py -—cov=lexbor.loss_metric_funcs

Or to save as html report:
$ pytest tests/test_loss_metric_funcs.py -—cov=lexbor.loss_metric_funcs --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_tloss_metric_funcs.py --cov=lexbor.loss_metric_funcs --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_loss_metric_funcs.py::test_x --cov=lexbor.loss_metric_funcs --cov-report term-missing
$ pytest tests/test_tloss_metric_funcs.py::test_y --cov=lexbor.loss_metric_funcs --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

Add -s to capture output.

"""
# import statistics
import pytest


# Import for development of model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.config.list_physical_devices("GPU")

import lexbor.loss_metric_funcs as lmf


"""
tf.keras.losses.SparseCategoricalCrossentropy()

In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Standalone usage:

  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
  >>> scce(y_true, y_pred).numpy()
  1.177
"""
def gather_preds_for_labels(label, pred):
    print(label)
    print(tf.shape(label))
    print(tf.shape(pred))
    # label (None, 32)  - None is batch, 32 is segments.
    # pred (None, 32, |vocab|) - None is batch, 32 is segments, |vocab|.
    # Need to index on batch and segments.

    index = tf.range(tf.shape(label)[-1])
    print("index:", index)
    index_2d = tf.stack([index, tf.cast(label, tf.int32)], axis=1)
    print("index2D:", index_2d)
    return tf.gather_nd(pred, indices=index_2d)

def gather_preds_for_labels2(label, pred):
    label = tf.cast(label, tf.int32)
    print(label)
    pred = tf.cast(pred, tf.float32)
    print(pred)
    result = tf.gather(pred, label, axis=2, batch_dims=2)
    print("Positive dims:", result)
    return result

def test_gather():
    print("Test - gather preds for labels.")
    label = [1, 2, 0]
    pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [.05, .95, 0]]
    result = gather_preds_for_labels(label, pred)
    print("Result for preds:", result)

def test_gather2():
    print("Test - gather preds for labels.")
    # Batch size of 2, sequence size of 3.
    label = [[1, 2, 0], [2, 1, 1]]
    pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1], [.05, .95, 0]],
            [[0.80, 0.15, 0.05], [0.70, 0.20, 0.10], [0.05, 0.90, 0.05]]]
    result = gather_preds_for_labels2(label, pred)
    print("Result for preds:", result)


def test_sparse_loss():
    print("Test sparse categorical loss.")
    label = [1, 2, 0]
    pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1], [.05, .95, 0]]

    result = lmf.sparseCategoricalCELoss(label, pred, reduction="sum")
    # gather_preds_from_labels(label, pred)
    print("CELoss sum", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction="sum",
                                      zero_masking=True)
    print("CELoss sum ", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None)
    print("CELoss none", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None,
                                     zero_masking=True)
    print("CELoss none ", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None,
                                     label_smoothing=0.2,
                                     smoothing_policy="one")
    print("CELoss smoothing one", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None,
                                     zero_masking=True,
                                     label_smoothing=0.2,
                                     smoothing_policy="one")
    print("CELoss smoothing one", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None,
                                     label_smoothing=0.2,
                                     smoothing_policy="all")
    print("CELoss smoothing all", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, reduction=None,
                                     zero_masking=True,
                                     label_smoothing=0.2,
                                     smoothing_policy="all")
    print("CELoss smoothing all", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred)
    print("CELoss", result.numpy())
    assert result.numpy() == pytest.approx(1.783, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred, zero_masking=True)
    print("CELoss", result.numpy())
    assert result.numpy() == pytest.approx(1.177, abs=0.001)


    result = lmf.sparseCategoricalCELoss(label, pred,
                                     label_smoothing=0.2,
                                     smoothing_policy="one")
    print("CELoss smoothing one", result.numpy())
    # Expected result: 0.8*1.78+0.2*[-ln(0.05)-ln(0.9)-ln(0.95)]/3
    # 1.424 + 0.2*[3.0+0.10+0.05]/3
    # 1.424 + 0.2*1.05 = 1.634
    assert result.numpy() == pytest.approx(1.637, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred,
                                     zero_masking=True,
                                     label_smoothing=0.2,
                                     smoothing_policy="one")
    print("CELoss smoothing one", result.numpy())
    # Expected result: 0.8*1.18+0.2*[-ln(0.05)-ln(0.9)]/2
    # 0.944 + 0.2*[3.0+0.1]/2
    # 0.944 + 0.2*1.55 = 1.254
    assert result.numpy() == pytest.approx(1.252, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred,
                                     label_smoothing=0.2,
                                     smoothing_policy="all")
    print("CELoss smoothing all", result.numpy())
    # Expected result: 0.8*1.78 + 0.2*[-ln(0.05/2)-ln(0.9/2)-ln(0.95/2))]/3
    # 1.424 + 0.2*[3.69+0.80+0.80]/3
    # 1.424 + 0.2*1.763 = 1.777
    assert result.numpy() == pytest.approx(1.775, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred,
                                     zero_masking=True,
                                     label_smoothing=0.2,
                                     smoothing_policy="all")
    print("CELoss smoothing all", result.numpy())
    # Expected result: 0.8*1.177 + 0.2*[-ln(0.05/1)-ln(0.9/1)]/3
    # 0.942 + 0.2*[2.996+0.105]/2
    # 0.942 + 0.2*1.550 = 1.252
    assert result.numpy() == pytest.approx(1.252, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred,
                                     label_smoothing=0.2,
                                     smoothing_policy="both")
    print("CELoss smoothing both", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred,
                                     zero_masking=True,
                                     label_smoothing=0.2,
                                     smoothing_policy="both")
    print("CELoss smoothing both", result.numpy())

    pred = [[-3.0, -.05, -10.0], [-2.3, -0.22, -2.3], [-3.0, -.05, -10.0]]
    result = lmf.sparseCategoricalCELoss(label, pred, from_logits=True)
    print("CELoss logits", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, from_logits=True,
                                      zero_masking=True)
    print("CELoss logits", result.numpy())

    result = lmf.sparseCategoricalCELoss(label, pred, from_logits=True,
                                      label_smoothing=0.2,
                                      smoothing_policy="one")
    print("CELoss logits smoothing one", result.numpy())
    assert result.numpy() == pytest.approx(1.638, abs=0.001)

    result = lmf.sparseCategoricalCELoss(label, pred, from_logits=True,
                                      zero_masking=True,
                                      label_smoothing=0.2,
                                      smoothing_policy="one")
    print("CELoss logits smoothing one", result.numpy())
    assert result.numpy() == pytest.approx(1.252, abs=0.001)

    with pytest.raises(ValueError):
        result = lmf.sparseCategoricalCELoss(label, pred, from_logits=True,
                                          zero_masking=True,
                                          label_smoothing=0.2,
                                          smoothing_policy="error")
        print("CELoss logits smoothing one", result.numpy())
