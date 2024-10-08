#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:26:16 2024

@author: johnmiller

Functions to calculate losses and metrics.
"""

import tensorflow as tf

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

Duplicate functionality of standard loss function.
Assume from probability distribution to start.
Smoothing policies of: None, one, all, both
"""

def sparseCategoricalCELoss(label, pred, reduction="mean",
                            from_logits=False, zero_masking=False,
                            label_smoothing=0.0, smoothing_policy=None):

    # loss vector: loss = -ln(pred[label]).
    # Reduction over batch: loss = mean(loss).
    # May be from logits instead of SoftMax prob estimates.
    # Assume 0 value for label indicates padding.

    if from_logits == True: # Get SoftMax estimates.
        pred = tf.nn.softmax(pred)

    pred_ = gather_preds_for_labels(label, pred)
    loss = tf.math.negative(tf.math.log(pred_))

    # Smoothing goes here.
    if label_smoothing > 0 and smoothing_policy is not None:
        epsilon = label_smoothing
        match smoothing_policy:
            case "one":
                loss1 = tf.math.negative(tf.math.log(tf.subtract(1.0, pred_)))
                loss = (1.0-epsilon)*loss + epsilon*loss1
            case "all":
                k = tf.shape(pred)[-1]
                k = tf.cast(k-2 if zero_masking else k-1, tf.float32)
                loss2 = tf.math.negative(tf.math.log(tf.subtract(1.0, pred_)/k))
                loss = (1.0-epsilon)*loss + epsilon*loss2
            case "both":
                loss1 = tf.math.negative(tf.math.log(tf.subtract(1.0, pred_)))
                k = tf.shape(pred)[-1]
                k = tf.cast(k-2 if zero_masking else k-1, tf.float32)
                loss2 = tf.math.negative(tf.math.log(tf.subtract(1.0, pred_)/k))
                loss = (1.0-epsilon)*loss + (epsilon/2.0)*(loss1+loss2)
            case _: raise ValueError(
                "Smoothing policy options: None, one, all, both. " +
                f" Not {smoothing_policy}.")


    if zero_masking:
        #print("Label", label)
        # Masking of label zero values.
        mask = tf.math.not_equal(label, 0)
        #print("Mask", mask)
        mask = tf.cast(mask, dtype=loss.dtype)
        #print("Mask", mask.numpy())
        loss *= mask
        #print("Loss", loss.numpy())

    match reduction:
        case "mean":
            if zero_masking:
                celoss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
            else:
                celoss = tf.reduce_mean(loss)
        case "sum": celoss = tf.reduce_sum(loss)
        case "None": celoss = loss
        case _: celoss = loss

    return celoss

def gather_preds_for_labels(label, pred):
    # print(label)
    # print(tf.shape(label))
    # print(tf.shape(pred))
    # label (None, 32)  - None is batch, 32 is segments.
    # pred (None, 32, |vocab|) - None is batch, 32 is segments, |vocab|.
    # Need to index on batch and segments.

    # index = tf.range(tf.shape(label)[-1])
    # index_2d = tf.stack([index, tf.cast(label, tf.int32)], axis=1)
    # return tf.gather_nd(pred, indices=index_2d, batch_dims=1)

    label = tf.cast(label, tf.int32)
    pred = tf.cast(pred, tf.float32)
    result = tf.gather(pred, label, axis=2, batch_dims=2)
    return result
