"""
Neural net entropy estimation module to configure and fit a neural network model

Notes
-----
Created on Tue Oct 3, 2020

@author: John E. Miller

Neural net entropy estimation module to configure and fit a neural network model, and
estimate entropies given a token or list of tokens. Tokens are encoded as lists of
integer ids where each id corresponds to a symbol segment from a vocabulary.

Updated on Nov 10, 2020 with corrected loss and metric functions and multihead attention.

@author Franco Pariasca - corrected loss and metric functions, and separate multihead attention.
@author John E. Miller - added classes for NeuralEntropyRecurrentAttention and
NeuralEntropyMultiheadAttention.
"""
# Import Python standard libraries
from pathlib import Path
# import math
# import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn
from keras.regularizers import l2

# Import 3rd-party libraries
# import abc
import attr

# Import tensorflow
# from keras.layers import Concatenate
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.layers import Input
# from keras.layers import LayerNormalization
from keras.layers import Concatenate, Add
from keras.layers import Attention
from keras.layers import Softmax
from keras import layers
from keras.models import Model
# from keras.regularizers import l2
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow import clip_by_value

# from tensorflow.python.ops import embedding_ops
# Changed to softmax in model final layer.
# No longer need to add softmax layer in prediction.
# This was necessary to prevent error in use of tensorflow 2.4.
# from keras import Sequential

# from keras import backend as K

# Build our namespace
# import pybor.neural_data as neu_dat
from lexbor import util
import lexbor.config as cfg
import lexbor.neural_entropy as entropy

"""
Neural net entropy estimation module to configure and fit a neural network model.
Recurrent attention language model; not just simple entropy calculation.

Notes
-----
Created on Wed Apl 28, 2021

@author: John E. Miller

Neural net entropy estimation module to configure and fit a neural network model, and
estimate entropies given a token or list of tokens. Tokens are encoded as lists of
integer ids where each id corresponds to a symbol segment from a vocabulary.
"""

output_path = Path(cfg.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)

EPSILON = 1e-7  # Used with prediction for clipping.


class MatrixMultiplyJEM(layers.Layer):

    def __init__(self, **kwargs):
        super(MatrixMultiplyJEM, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        transform = inputs[0]
        x = inputs[1]
        y = tf.transpose(x, perm=[0, 2, 1])
        y = tf.matmul(transform, y)
        y = tf.transpose(y, perm=[0, 2, 1])
        return y


# Forward only version - revised Sep 3, 2021.
# Distinguish source from target.
# Preparation for translation model.
@attr.s
class NeuralEntropyRecurrentLMFor(entropy.NeuralEntropy):

    translation = attr.ib(default=False)  # Reconstitution model is default

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Construct layer objects.
        # Single character segment input per prediction.
        # Variable length sequences.
        inputs = Input(shape=(None,), name="Segment_input")
        if self.translation:
            inputs_target = Input(shape=(None,), name="Target_segment_input")
        else:
            inputs_target = inputs

        # Embedding of segments.
        # Shared between forward, backward,
        # Shared with decoding and output layers if reconstitution.
        embedding = Embedding(
            input_dim=self.vocab_len,
            output_dim=params.embedding_len,
            mask_zero=True,
            name="Segment_embedding"
        )
        if self.translation:
            target_embedding = Embedding(
                input_dim=self.vocab_len,
                output_dim=params.embedding_len,
                mask_zero=True,
                name="Target_segment_embedding"
            )
        else:
            # Target and source embedding are the same.
            target_embedding = embedding

        embedding_dropout = Dropout(
            params.embedding_dropout,
            name="Embedding_dropout")

        # Forward lstm layers. No sharing.
        rnn_encoding_for = LSTM(
            params.rnn_output_len,
            kernel_regularizer=l2(params.kernel_l2),
            return_sequences=True,
            return_state=False,
            name="Recurrent_encoding")

        rnn_decoding_for = LSTM(
            params.rnn_output_len,
            return_sequences=True,
            kernel_regularizer=l2(params.kernel_l2),
            return_state=False,
            name="Recurrent_decoding")

        # Add encoding and inputs between this and the attention layer.
        # value_encode = tf.math.add(embedded_inputs,
        # rnn_hidden_encode, name="Calc_value_encode")

        # Shared attention layer.
        mul_attention = Attention(
            use_scale=True, use_causal_mask=params.attention_causal,
            dropout=params.attention_dropout,
            name="Mul_attention")

        # Add attention contents and rnn_hidden_encode
        # value_decode = tf.math.add(attn_contents,
        # rnn_hidden_decode, name="Calc_value_decode")

        # Shared dropout layer.
        decoding_dropout = Dropout(
            params.decoding_dropout,
            name="Dropout_decoding")

        # Shared linear decoding layer.
        decoding_linear_trans = Dense(
            params.rnn_output_len,
            use_bias=False,
            name="Decoding_linear_transform")

        # Shared pre-multiply by embedding layer.
        mul_embedding_decoding = MatrixMultiplyJEM(
            name="Mul_embedding_decoding")

        # Shared dropout layer.
        logit_dropout = Dropout(
            params.logit_dropout,
            name="Logit_dropout")

        # Shared Softmax layer.
        softmax_decoding = Softmax()

        # *** Construct the data flow between layers ***

        # *** Forward direction ***
        embedded_inputs_source = embedding(inputs)
        embedded_inputs_target = target_embedding(inputs_target)
        embedded_inputs_source_do = embedding_dropout(embedded_inputs_source)
        embedded_inputs_target_do = embedding_dropout(embedded_inputs_target)

        # Question: should same embedding after dropout be used for encoder and decoder?
        # Question: Or should dropouts be independent?
        rnn_encoded_seq = rnn_encoding_for(embedded_inputs_source_do)
        rnn_decoded_seq = rnn_decoding_for(embedded_inputs_target_do)

        value_encoded_seq = tf.math.add(
            embedded_inputs_source_do,
            rnn_encoded_seq,
            name="Calc_encoded_seq")
        # Query from decoder, value encoder and embeddings, key from encoder
        attn_contents = mul_attention(
            [rnn_decoded_seq,
             value_encoded_seq,
             rnn_encoded_seq])

        # Sum contents decoded sequence and linearly transform.
        # value_decoded_seq = tf.math.add(
        #     attn_contents,
        #     rnn_decoded_seq,
        #     name="Calc_decoded_seq")
        value_decoded_seq = Add()([attn_contents, rnn_decoded_seq])
        value_decoded_seq = decoding_dropout(value_decoded_seq)
        value_decoded_seq = decoding_linear_trans(value_decoded_seq)

        # embeddings is V x E (vocab_len x embedding_len).
        # to_outputs is B x T x H (batch_size, time_len, hidden_len) of logits.
        # H and E must be equal for multiply.
        # Embeddings are broadcast for batch and time dimensions.
        logit_outputs = mul_embedding_decoding(
            [target_embedding.embeddings, value_decoded_seq])
        if params.logit_dropout > 0:
            logit_outputs = logit_dropout(logit_outputs)

        outputs = softmax_decoding(logit_outputs)
        # Use of embeddings replaces the dense layer.

        model_name = self.construct_modelname("RecurrentAttentionLMFor")
        if not self.translation:
            self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)
        else:
            self.model = Model(inputs=[inputs, inputs_target], outputs=[outputs], name=model_name)

        if params.verbose > 1:
            self.model.summary()
        if params.print_summary > 0:
            self.print_model_summary()


@attr.s
class NeuralEntropyRecurrentLMForBack(entropy.NeuralEntropy):

    translation = attr.ib(default=False)  # Reconstitution model is default

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Construct layer objects.
        # Single character segment input per prediction.
        # Variable length sequences.
        inputs = Input(shape=(None,), name="Segment_input")
        if self.translation:
            inputs_target = Input(shape=(None,), name="Target_segment_input")
        else:
            inputs_target = inputs

        # Embedding of segments.
        # Shared between forward, backward,
        # Shared with decoding and output layers if reconstitution.
        embedding = Embedding(
            input_dim=self.vocab_len,
            output_dim=params.embedding_len,
            mask_zero=True,
            name="Segment_embedding"
        )
        if self.translation:
            target_embedding = Embedding(
                input_dim=self.vocab_len,
                output_dim=params.embedding_len,
                mask_zero=True,
                name="Target_segment_embedding"
            )
        else:
            # Target and source embedding are the same.
            target_embedding = embedding

        embedding_dropout = Dropout(
            params.embedding_dropout,
            name="Embedding_dropout")

        # Forward lstm layers. No sharing.
        rnn_encoding_for = LSTM(
            params.rnn_output_len,
            kernel_regularizer=l2(params.kernel_l2),
            return_sequences=True,
            return_state=False,
            name="Recurrent_encoding")

        rnn_decoding_for = LSTM(
            params.rnn_output_len,
            kernel_regularizer=l2(params.kernel_l2),
            return_sequences=True,
            return_state=False,
            name="Recurrent_decoding")

        # Backward lstm layers. No sharing.
        rnn_encoding_back = LSTM(
            params.rnn_output_len,
            kernel_regularizer=l2(params.kernel_l2),
            return_sequences=True,
            return_state=False,
            go_backwards=True,
            name="Recurrent_encoding_back")

        rnn_decoding_back = LSTM(
            params.rnn_output_len,
            kernel_regularizer=l2(params.kernel_l2),
            return_sequences=True,
            return_state=False,
            go_backwards=True,
            name="Recurrent_decoding_back")

        # Add encoding and inputs between this and the attention layer.
        # value_encode = tf.math.add(embedded_inputs,
        # rnn_hidden_encode, name="Calc_value_encode")

        # Shared attention layer.
        mul_attention = Attention(
            use_scale=True, use_causal_mask=params.attention_causal,
            dropout=params.attention_dropout,
            name="Mul_attention")

        # Add attention contents and rnn_hidden_encode
        # value_decode = tf.math.add(attn_contents,
        # rnn_hidden_decode, name="Calc_value_decode")

        # Shared dropout layer.
        decoding_dropout = Dropout(
            params.embedding_dropout,
            name="Dropout_decoding")

        # Shared linear decoding layer.
        decoding_linear_trans = Dense(
            params.rnn_output_len,
            use_bias=False,
            name="Decoding_linear_transform")

        # Shared pre-multiply by embedding layer.
        mul_embedding_decoding = MatrixMultiplyJEM(
            name="Mul_embedding_decoding")

        # Shared dropout layer.
        logit_dropout = Dropout(
            params.logit_dropout,
            name="Logit_dropout")

        # Shared Softmax layer.
        softmax_decoding = Softmax()

        # *** Construct the data flow between layers ***
        # *** Construct the data flow between layers ***

        # *** Forward direction encoder and decoder ***
        embedded_inputs_source = embedding(inputs)  # (1)
        embedded_inputs_target = target_embedding(inputs_target)  # (3)
        embedded_inputs_source_do_for = embedding_dropout(embedded_inputs_source)  # (1')
        embedded_inputs_target_do_for = embedding_dropout(embedded_inputs_target)  # (3')
        rnn_encoded_seq_for = rnn_encoding_for(embedded_inputs_source_do_for)  # (2)
        rnn_decoded_seq_for = rnn_decoding_for(embedded_inputs_target_do_for)  # (3.2)

        # *** Backward direction encoder and decoder ***
        embedded_inputs_source_do_back = embedding_dropout(embedded_inputs_source)  # (1')
        embedded_inputs_target_do_back = embedding_dropout(embedded_inputs_target)  # (3')
        rnn_encoded_seq_back = rnn_encoding_back(embedded_inputs_source_do_back)  # (2)
        rnn_decoded_seq_back = rnn_decoding_back(embedded_inputs_target_do_back)  # (3.4)

        # *** Attention mechanism shared across forward and backward ***
        # HOW???
        value_encoded_seq_for = tf.math.add(  # (6)
            embedded_inputs_source_do_for,
            rnn_encoded_seq_for,
            name="Calc_encoded_seq_for")
        value_encoded_seq_back = tf.math.add(  # (6)
            embedded_inputs_source_do_back,
            rnn_encoded_seq_back,
            name="Calc_encoded_seq_back")

        # Query from decoder, value encoder and embeddings, key from encoder
        # Concatenate inputs for attention.  Maybe dependency, we'll see.

        value_encoded_seq = Concatenate()([value_encoded_seq_for, value_encoded_seq_back])
        rnn_encoded_seq = Concatenate()([rnn_encoded_seq_for, rnn_encoded_seq_back])
        rnn_decoded_seq = Concatenate()([rnn_decoded_seq_for, rnn_decoded_seq_back])
        attn_contents = mul_attention(  # (6.1)
            [rnn_decoded_seq,
             value_encoded_seq,
             rnn_encoded_seq])

        # Sum contents decoded sequence and linearly transform.
        # value_decoded_seq = tf.math.add(  # (5)
        #     attn_contents,
        #     rnn_decoded_seq,
        #     name="Calc_decoded_seq")
        value_decoded_seq = Add()([attn_contents, rnn_decoded_seq])
        value_decoded_seq = decoding_dropout(value_decoded_seq)
        value_decoded_seq = decoding_linear_trans(value_decoded_seq)  # (5)

        # embeddings is V x E (vocab_len x embedding_len).
        # to_outputs is B x T x H (batch_size, time_len, hidden_len) of logits.
        # H and E must be equal for multiply.
        # Embeddings are broadcast for batch and time dimensions.
        logit_outputs = mul_embedding_decoding(
            [target_embedding.embeddings, value_decoded_seq])
        if params.logit_dropout > 0.0:
            logit_outputs = logit_dropout(logit_outputs)

        outputs = softmax_decoding(logit_outputs)

        model_name = self.construct_modelname("RecurrentAttentionLMForBack")
        if not self.translation:
            self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)
        else:
            self.model = Model(inputs=[inputs, inputs_target], outputs=[outputs], name=model_name)

        if params.verbose > 1:
            self.model.summary()
        if params.print_summary > 0:
            self.print_model_summary()
