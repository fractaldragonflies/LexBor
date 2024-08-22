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
import unidecode
import math
# import numpy as np

# Import 3rd-party libraries
import abc
import attr

# Import tensorflow
import tensorflow as tf
from keras.callbacks import EarlyStopping  # ModelCheckpoint,
from keras.layers import Concatenate
from keras.layers import Dense, Embedding, Dropout
from keras.layers import GRU, LSTM
from keras.layers import Input
from keras.layers import LayerNormalization
from keras.layers import Attention, AdditiveAttention
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import pad_sequences
from keras.regularizers import l2
from keras.utils import plot_model
from tensorflow import clip_by_value
# Changed to softmax in model final layer.
# No longer need to add softmax layer in prediction.
# This was necessary to prevent error in used of tensorflow 2.4.
# from keras import Sequential

# from keras import backend as K

# Build our namespace
import lexbor.util as util
import lexbor.config as cfg

import lexbor.multihead_attention as mha

# @title src/pybor/entropies_with_attention_modified.py
# %%writefile src/pybor/entropies_with_attention_modified.py
# ############################Loss sparse_categorical_crossentropy#############################


def loss_sparse_categorical_crossentropy(y_true, y_pred):
    loss_function = tf.keras.losses.sparse_categorical_crossentropy

    # loss = loss_function(y_true, y_pred, from_logits=True)
    loss = loss_function(y_true, y_pred, from_logits=False)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))

    mask = tf.cast(mask, dtype=loss.dtype)

    loss *= mask

    return tf.reduce_mean(tf.reduce_sum(loss) / tf.reduce_sum(mask))

# ############################Loss sparse_categorical_crossentropy#############################
# ############################Metric sparse_categorical_accuracy#############################


def metric_sparse_categorical_accuracy(y_true, y_pred):
    metric_function = tf.keras.metrics.sparse_categorical_accuracy

    metric = metric_function(y_true, y_pred)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))

    mask = tf.cast(mask, dtype=metric.dtype)

    metric *= mask

    return tf.reduce_mean(tf.reduce_sum(metric) / tf.reduce_sum(mask))

# ############################Metric sparse_categorical_accuracy#############################
# ############################Metric sparse_categorical_accuracy#############################


def metric_sparse_categorical_crossentropy(y_true, y_pred):
    metric_function = tf.keras.metrics.sparse_categorical_crossentropy

    # metric = metric_function(y_true, y_pred, from_logits=True)
    metric = metric_function(y_true, y_pred, from_logits=False)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))

    mask = tf.cast(mask, dtype=metric.dtype)

    metric *= mask

    return tf.reduce_mean(tf.reduce_sum(metric) / tf.reduce_sum(mask))


"""
Neural net entropy estimation module to configure and fit a neural network model

Notes
-----
Created on Tue Oct 3, 2020

@author: John E. Miller

Neural net entropy estimation module to configure and fit a neural network model, and
estimate entropies given a token or list of tokens. Tokens are encoded as lists of
integer ids where each id corresponds to a symbol segment from a vocabulary.
"""


output_path = Path(cfg.BaseSettings().output_path).resolve()
logger = util.get_logger(__name__)

EPSILON = 1e-7  # Used with prediction for clipping.


@attr.s
class NeuralEntropy:
    """
    Use lists of token id lists to calculate token entropies.
    Neural net model is configured and then trained from a list of tokens at
    time of construction. Entropy or entropy lists are calculated from the
    trained neural net model.
    """
    """
    Neural net based model to calculate token entropy.
    Configure and compile the model and fit training data at construction.
    Calculate entropies on demand based on the fitted model.

    Parameters
    ----------
    vocab_len : int
        Length of the vocabulary. Note: All token segments are integer encoded.
    language : str
        Language being modeled. Used in model naming for storage.
    basis : str ['all', 'native', 'loan', ...]
        Whether all tokens, just native, or just loan.
    series : str
        Study series to qualify model name.

    Returns
    -------
    NeuralEntropy object reference.

    Notes
    -----
    Based on recent research, dropout of 0.1 should be considered when model includes
    multiple levels of dropout.
    """

    __metaclass__ = abc.ABCMeta

    vocab_len = attr.ib(default=None)
    language = attr.ib(default=None)
    basis = attr.ib(default=None)
    series = attr.ib(default=None)
    settings = attr.ib(default=cfg.RecurrentSettings())

    model = attr.ib(init=False)
    prob_model = attr.ib(init=False)

    train_measures = attr.ib(init=False, default=None)
    test_measures = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):

        if self.vocab_len is None or self.vocab_len <= 0:
            raise ValueError(
                "Require a vocabulary size > 0 to construct neural entropy model."
            )
        self.language = self.settings.language if self.language is None else self.language
        self.basis = self.settings.basis if self.basis is None else self.basis
        self.series = self.settings.series if self.series is None else self.series

    def calculate_entropy(self, token_ids):
        """
        Compute the entropy for a given token.
        """

        return self.calculate_entropies([token_ids])[0]

    def calculate_entropies(self, tokens_ids):
        """
        Compute the entropy for a collection of tokens.
        """

        assert tokens_ids is not None and len(tokens_ids) > 0
        #    # Calculate entropy for a list of tokens_ids
        # in format of int ids that correspond to str segments.
        # Get the probabilities for all str segment possibilities.
        maxlen = max([len(token_ids) for token_ids in tokens_ids])
        # Truncate right id for x and left id for y, so only 1 id extra.
        if maxlen > self.settings.token_maxlen + 1:
            maxlen = self.settings.token_maxlen + 1

        x_lst = []
        y_lst = []
        for token_ids in tokens_ids:
            x_lst.append(token_ids[:-1])
            y_lst.append(token_ids[1:])

        x_tf = pad_sequences(x_lst, padding="post", maxlen=maxlen)
        y_probs = self.model(x_tf)

        # Compute cross-entropies

        entropies = []
        for y_ids_probs, y_ids in zip(y_probs, y_lst):
            # Prevent overflow/underflow with clipping.
            y_ids_probs_ = clip_by_value(y_ids_probs,  EPSILON, 1-EPSILON)
            y_ids_lns = [
                math.log(y_ids_probs_[i, y_ids[i]])
                for i in range(min(maxlen, len(y_ids)))
            ]
            entropy = -sum(y_ids_lns) / math.pow(len(y_ids_lns), self.settings.len_power)
            entropies.append(entropy)

        assert len(tokens_ids) == len(entropies)
        return entropies

    def train(self, train_gen=None, val_gen=None, epochs=None):
        """
        Train the neural network using training and validation data in generators.

        Parameters
        ----------
        train_gen : Keras generator
            Training generator. The default is None.
        val_gen : Keras generator
            Validation generator. The default is None.
        epochs : int
            Number epochs training, overrides config. The default is None.

        Returns
        -------
        Tensorflow history.history.

        Notes
        -----
            Invoke train after construction of the model.
            It's too heavy weight to do in init of class.

        """

        if train_gen is None:
            logger.error("There is no training data for train()")
            raise ValueError("Require training data to train the entropy model.")

        if self.settings.verbose > 0:
            logger.info("Training neural %s model.", str(type(self)))

        learning_rate = self.settings.learning_rate
        learning_rate_decay = self.settings.learning_rate_decay
        # Without min of 1, system uses 0 which is an error.
        # If result > 0 system will adjust up for remnant of batch.
        train_steps = max(1, (train_gen.data_len // train_gen.batch_size))
        epochs = epochs if epochs is not None else self.settings.epochs

        # Convert this to learning rate schedule.
        lr_schedule = self.settings.learning_rate_schedule
        if lr_schedule is None:
            # Use Adam built in adjustment.
            optimizer = Adam(learning_rate=learning_rate)
        elif lr_schedule == 'EX':
            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=train_steps, decay_rate=learning_rate_decay)
            optimizer = Adam(learning_rate=learning_rate_schedule)
        elif lr_schedule == 'IT':
            learning_rate_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                learning_rate, decay_steps=train_steps, decay_rate=(1.0-learning_rate_decay))
            optimizer = Adam(learning_rate=learning_rate_schedule)
        elif lr_schedule == 'CU':
            learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                learning_rate, decay_steps=train_steps*epochs,
                end_learning_rate=0.1*learning_rate, power=3)
            optimizer = Adam(learning_rate=learning_rate_schedule)
        elif lr_schedule == 'TR':
            learning_rate_schedule = TRSchedule(model_dim=self.settings.embedding_len,
                                                warmup=self.settings.learning_rate_warmup,
                                                learning_rate=learning_rate)
            optimizer = Adam(learning_rate=learning_rate_schedule)
        else:  # Use previous learning rate schedule
            learning_rate_decay = self.settings.learning_rate_decay
            # Transform to per step decay.
            learning_rate_decay = (1.0 / learning_rate_decay - 1.0) / train_steps
            if self.settings.verbose > 0:
                logger.info("Using per step learning rate decay %.4f", learning_rate_decay)
            optimizer = Adam(learning_rate=learning_rate, decay=learning_rate_decay)

        self.model.compile(
            loss=loss_sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=[
                metric_sparse_categorical_accuracy,
                metric_sparse_categorical_crossentropy],
        )

        callbacks = []
        if self.settings.early_stopping and val_gen:
            # Early stopping monitors validation measure.
            earlystopper = EarlyStopping(
                monitor="metric_sparse_categorical_crossentropy",
                verbose=self.settings.tf_verbose,
                patience=self.settings.patience,
                restore_best_weights=self.settings.restore_best_weights
            )
            callbacks = [earlystopper]

        if val_gen:
            val_steps = max(1, (val_gen.data_len // val_gen.batch_size))
            history = self.model.fit(
                train_gen.generate(),
                steps_per_epoch=train_steps,
                epochs=epochs,
                validation_data=val_gen.generate(),
                validation_steps=val_steps,
                verbose=self.settings.tf_verbose,
                callbacks=callbacks,
            )
        else:
            history = self.model.fit(
                train_gen.generate(),
                steps_per_epoch=train_steps,
                epochs=epochs,
                verbose=self.settings.tf_verbose,
            )

        if self.settings.verbose > 0:
            self.show_quality_measures(history.history)

        return history.history

    def show_quality_measures(self, history):
        """
        Report quality measures.
        """

        history_keys = history.keys()
        if not self.train_measures:
            logger.info(f"Available quality measures: {history_keys}.")
            self.evaluate_quality_measures(history)

        measures = self.train_measures
        logger.info("Statistics from TensorFlow:")
        logger.info(
            f"Train dataset: loss={measures[0]:.4f}, "
            + f"accuracy={measures[1]:.4f}, "
            + f"cross_entropy={measures[2]:.4f}."
        )
        if "val_loss" in history_keys:
            logger.info(
                f"Validate dataset: loss={measures[3]:.4f}, "
                + f"accuracy={measures[4]:.4f}, "
                + f"cross_entropy={measures[5]:.4f}."
            )
        else:
            logger.info("No validation results reported.")

    def evaluate_quality_measures(self, history):
        history_keys = history.keys()
        if ("val_metric_sparse_categorical_crossentropy" in history_keys and
                self.settings.early_stopping and self.settings.restore_best_weights):
            measure = history["val_metric_sparse_categorical_crossentropy"]
            idx, best_measure = min(enumerate(measure), key=lambda v: v[1])
            if self.settings.verbose > 1:
                logger.info(f"Restore best epoch: {idx} of {len(measure)}.")
        else:
            idx = -1

        measures = [history['loss'][idx]]
        measures += [history['metric_sparse_categorical_accuracy'][idx]]
        measures += [history['metric_sparse_categorical_crossentropy'][idx]]
        if "val_loss" in history_keys:
            measures += [history['val_loss'][idx]]
            measures += [history['val_metric_sparse_categorical_accuracy'][idx]]
            measures += [history['val_metric_sparse_categorical_crossentropy'][idx]]

        self.train_measures = measures

    def show_test(self, test_gen=None):
        if not self.test_measures:
            if test_gen:
                self.evaluate_test(test_gen)

        if self.test_measures:
            logger.info(
                f"Test dataset: loss={self.test_measures[0]:.4f}, "
                + f"accuracy={self.test_measures[1]:.4f}, cross_entropy={self.test_measures[2]:.4f}."
                )

    def evaluate_test(self, test_gen=None):
        # Evaluate using generator - use evaluate directly.
        if not test_gen:
            logger.warning("No test data for evaluation!")
            return

        test_steps = max(1, (test_gen.data_len // test_gen.batch_size))
        self.test_measures = self.model.evaluate(
            test_gen.generate(), steps=test_steps, verbose=self.settings.tf_verbose
        )

    def print_model_summary(self):
        # Prints model summary to log path.
        print_fn = output_path / (self.model.name + ".txt")
        with open(print_fn.as_posix(), "w") as handler:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: handler.write(x + "\n"))

    def plot_model_summary(self):
        print_fn = output_path / (self.model.name + ".png")
        plot_model(
            self.model,
            print_fn.as_posix(),
            show_shapes=True,
            show_layer_names=True,
            dpi=self.settings.plot_dpi,
        )

    def construct_modelprefix(self):
        language_out = "".join(self.language.split())
        language_out = unidecode.unidecode(language_out)
        language_out = language_out.replace("'", "")

        model_prefix = f"{language_out}-" if language_out != "" else ""
        model_prefix += f"{self.basis}-" if self.basis != "" else ""
        model_prefix += f"{self.series}-" if self.series != "" else ""
        model_prefix = model_prefix.replace(" ", "_")
        return model_prefix

    def construct_modelsuffix(self):
        embedding_len = self.settings.embedding_len
        rnn_output_len = self.settings.rnn_output_len
        rnn_cell_type = self.settings.rnn_cell_type
        model_suffix = (
            f"-emblen{embedding_len}-rnnlen{rnn_output_len}-celtyp{rnn_cell_type}"
        )
        return model_suffix

    def construct_modelname(self, model_type):
        return self.construct_modelprefix() + model_type + self.construct_modelsuffix()

    @abc.abstractmethod
    def build_model(self):
        ...

    def get_embedding_weights(self):
        return self.model.get_layer('Segment_embedding').get_weights()[0]


@attr.s
class NeuralEntropyRecurrent(NeuralEntropy):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,), name="Segment_input")

        # Embedding of characters.
        # Mask zero works.
        embedding = Embedding(
            input_dim=self.vocab_len,
            output_dim=params.embedding_len,
            mask_zero=True,
            name="Segment_embedding",
        )(inputs)

        # Retain reference to embeddings without dropout.
        embedding_for_merge = embedding

        if params.embedding_dropout > 0.0:
            embedding = Dropout(params.embedding_dropout,
                                name="Dropout_embedding")(embedding)

        if params.rnn_cell_type == "LSTM":
            # Incorporate embeddings into hidden state and output state.
            rnn_output = LSTM(
                params.rnn_output_len,
                return_sequences=True,
                return_state=False,
                recurrent_regularizer=l2(params.recurrent_l2),
                activity_regularizer=l2(params.rnn_activity_l2),
                recurrent_dropout=params.recurrent_dropout,
                name="LSTM_recurrent",
            )(embedding)

        else:  # GRU
            rnn_output = GRU(
                params.rnn_output_len,
                return_sequences=True,
                return_state=False,
                recurrent_regularizer=l2(params.recurrent_l2),
                activity_regularizer=l2(params.rnn_activity_l2),
                recurrent_dropout=params.recurrent_dropout,
                name="GRU_recurrent",
            )(embedding)

        if params.rnn_levels == 2:
            if params.rnn_cell_type == "LSTM":
                rnn_output = LSTM(
                    params.rnn_output_len,
                    return_sequences=True,
                    return_state=False,
                    recurrent_regularizer=l2(params.recurrent_l2),
                    activity_regularizer=l2(params.rnn_activity_l2),
                    recurrent_dropout=params.recurrent_dropout,
                    name="LSTM_recurrent_2",
                )(rnn_output)

            else:  # GRU
                rnn_output = GRU(
                    params.rnn_output_len,
                    return_sequences=True,
                    return_state=False,
                    recurrent_regularizer=l2(params.recurrent_l2),
                    activity_regularizer=l2(params.rnn_activity_l2),
                    recurrent_dropout=params.recurrent_dropout,
                    name="GRU_recurrent_2",
                )(rnn_output)

        if params.rnn_output_dropout > 0.0:
            rnn_output = Dropout(params.rnn_output_dropout,
                                 name="Dropout_rnn_output")(rnn_output)

        if params.merge_embedding:
            # Add in embedding per Bengio 2002.
            if params.merge_embedding_dropout > 0.0:
                embedding_for_merge = Dropout(params.merge_embedding_dropout,
                                              name='merge_emb_do')(embedding_for_merge)
            rnn_output = Concatenate(axis=-1, name="Merge_rnn_embedding")(
                                         [rnn_output, embedding_for_merge])

        # Hidden state used to predict subsequent character.
        # outputs = Dense(self.vocab_len, name="Segment_output")(rnn_output)
        outputs = Dense(self.vocab_len, activation='softmax',
                        name="Segment_output")(rnn_output)

        model_name = self.construct_modelname("Recurrent")
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if params.print_summary > 0:
            self.print_model_summary()
        if params.plot_model > 0:
            self.plot_model_summary()

        if params.verbose > 1:
            self.model.summary()


@attr.s
class NeuralEntropyRecurrentAttention(NeuralEntropy):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,), name="Segment_input")

        # Embedding of characters.
        # Mask zero works.
        embedding = Embedding(
            input_dim=self.vocab_len,
            output_dim=params.embedding_len,
            mask_zero=True,
            name="Segment_embedding",
        )(inputs)

        if params.embedding_dropout > 0.0:
            embedding = Dropout(params.embedding_dropout,
                                name="Dropout_embedding")(embedding)

        # Only use 1 layer with attention.  Doesn't seem effective anyways.
        if params.rnn_cell_type == "LSTM":
            # Incorporate embeddings into hidden state and output state.
            rnn_output = LSTM(
                params.rnn_output_len,
                return_sequences=True,
                return_state=False,
                recurrent_regularizer=l2(params.recurrent_l2),
                activity_regularizer=l2(params.rnn_activity_l2),
                recurrent_dropout=params.recurrent_dropout,
                name="LSTM_recurrent",
            )(embedding)

        else:  # GRU
            rnn_output = GRU(
                params.rnn_output_len,
                return_sequences=True,
                return_state=False,
                recurrent_regularizer=l2(params.recurrent_l2),
                activity_regularizer=l2(params.rnn_activity_l2),
                recurrent_dropout=params.recurrent_dropout,
                name="GRU_recurrent",
            )(embedding)

        if params.rnn_output_dropout > 0.0:
            rnn_output = Dropout(params.rnn_output_dropout,
                                 name="Dropout_rnn_output")(rnn_output)

        if params.attention_type == "ADD":
            to_outputs = AdditiveAttention(use_scale=True, causal=True,
                                           dropout=params.attention_dropout,
                                           name="Add_attention")([embedding, rnn_output])
        else:
            to_outputs = Attention(use_scale=True, causal=True,
                                   dropout=params.attention_dropout,
                                   name="Mul_attention")([embedding, rnn_output])

        if params.transformer_stuff:
            to_outputs = LayerNormalization(epsilon=1e-6,
                                            name="Transformer_attn_norm")(
                to_outputs + rnn_output)
            to_outputs_1 = Dense(params.rnn_output_len, activation='relu',
                                 name="Transformer_dense")(to_outputs)
            to_outputs_1 = Dropout(params.transformer_dropout,
                                   name="Transformer_dropout")(to_outputs_1)
            to_outputs = LayerNormalization(epsilon=1e-6,
                                            name="Transformer_norm")(
                to_outputs + to_outputs_1)

        if params.attention_levels == 2:
            layer_1_outputs = to_outputs

            if params.attention_type == "ADD":
                to_outputs = AdditiveAttention(use_scale=True, causal=True,
                                               dropout=params.attention_dropout,
                                               name="Add_attention_2")(
                    [rnn_output, layer_1_outputs])
            else:
                to_outputs = Attention(use_scale=True, causal=True,
                                       dropout=params.attention_dropout,
                                       name="Mul_attention_2")(
                    [rnn_output, layer_1_outputs])

            if params.transformer_stuff:
                to_outputs = LayerNormalization(epsilon=1e-6,
                                                name="Transformer_attn_norm_2")(
                    to_outputs + layer_1_outputs)
                to_outputs_1 = Dense(params.rnn_output_len, activation='relu',
                                     name="Transformer_dense_2")(to_outputs)
                to_outputs_1 = Dropout(params.transformer_dropout,
                                       name="Transformer_dropout_2")(to_outputs_1)
                to_outputs = LayerNormalization(epsilon=1e-6,
                                                name="Transformer_norm_2")(
                    to_outputs + to_outputs_1)

        if params.merge_embedding:
            to_outputs = Concatenate(axis=-1, name="Merge_attn_embedding")(
                 [to_outputs, embedding])

        # outputs = Dense(self.vocab_len, name="Segment_output")(to_outputs)
        outputs = Dense(self.vocab_len, activation='softmax',
                        name="Segment_output")(to_outputs)

        model_name = self.construct_modelname("RecurrentAttention")
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if params.print_summary > 0:
            self.print_model_summary()
        if params.plot_model > 0:
            self.plot_model_summary()

        if params.verbose > 1:
            self.model.summary()


@attr.s
class NeuralEntropyMultiheadAttention(NeuralEntropy):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.build_model()

    def build_model(self):
        params = self.settings  # Convenience

        # Single character segment input per prediction. Variable length sequences.
        inputs = Input(shape=(None,), name="Segment_input")

        # Embedding of characters.
        # Mask zero works.
        character_embeddings = Embedding(
            input_dim=self.vocab_len,
            output_dim=params.embedding_len,
            mask_zero=True,
            name="Segment_embedding",
        )(inputs)

        # Encoding of positions - either by embedding or by trigonometric function.
        max_len = tf.shape(inputs)[-1]
        if params.positional_encoding == 'EMB':
            # Embedding of positions
            positions = tf.range(start=0, limit=max_len, delta=1)
            positional_encodings = Embedding(input_dim=params.token_maxlen+1,
                                             output_dim=params.embedding_len,
                                             name="Positional_embedding")(positions)
        else:  # TRIG  # Trigonometric encoding
            positional_encodings = mha.positional_encoding(
                            params.token_maxlen+1, params.embedding_len)
            positional_encodings = positional_encodings[:, :max_len, :]

        embedding = character_embeddings + positional_encodings

        if params.embedding_dropout > 0.0:
            embedding = Dropout(params.embedding_dropout,
                                name="Embedding_dropout")(embedding)

        to_outputs = mha.MultiHeadSelfAttention(params.embedding_len,
                                                params.num_heads,
                                                True)(embedding)

        if params.attention_dropout > 0:
            to_outputs = Dropout(params.attention_dropout,
                                 name="Attention_dropout")(to_outputs)

        # To test in tf nightly build.
        # to_outputs = tf.keras.layers.MultiHeadAttention(params.num_heads, params.embedding_len,
        #                        dropout=params.attention_dropout, name="Self_attention",
        #                        causal=True)(embedding)

        if params.transformer_stuff:
            to_outputs = LayerNormalization(epsilon=1e-6,
                                            name="Transformer_attn_norm")(
                to_outputs + embedding)
            to_outputs_1 = Dense(params.embedding_len, activation='relu',
                                 name="Transformer_dense")(to_outputs)
            to_outputs_1 = Dropout(params.transformer_dropout,
                                   name="Transformer_dropout")(to_outputs_1)
            to_outputs = LayerNormalization(epsilon=1e-6,
                                            name="Transformer_norm")(
                to_outputs + to_outputs_1)

        if params.attention_levels == 2:
            layer_1_outputs = to_outputs

            to_outputs = mha.MultiHeadSelfAttention(params.embedding_len,
                                                    params.num_heads,
                                                    True)(layer_1_outputs)
            # Test attention from  tf nightly build.
            if params.attention_dropout > 0:
                to_outputs = Dropout(params.attention_dropout,
                                     name="Attention_dropout_2")(to_outputs)

            if params.transformer_stuff:
                to_outputs = LayerNormalization(epsilon=1e-6,
                                                name="Transformer_attn_norm_2")(
                    to_outputs + layer_1_outputs)

                to_outputs_1 = Dense(params.embedding_len, activation='relu',
                                     name="Transformer_dense_2")(to_outputs)
                to_outputs_1 = Dropout(params.transformer_dropout,
                                       name="Transformer_dropout_2")(to_outputs_1)
                to_outputs = LayerNormalization(epsilon=1e-6,
                                                name="Transformer_norm_2")(
                    to_outputs + to_outputs_1)

        if params.merge_embedding:
            to_outputs = Concatenate(axis=-1, name="Merge_attn_embedding")(
                [to_outputs, embedding])

        # Include softmax to see if tf 2.4 works correctly to predict probabilities.
        # outputs = Dense(self.vocab_len, name="Segment_output")(to_outputs)
        outputs = Dense(self.vocab_len, activation='softmax',
                        name="Segment_output")(to_outputs)

        model_name = self.construct_modelname("MultiheadAttention")
        self.model = Model(inputs=[inputs], outputs=[outputs], name=model_name)

        if params.print_summary > 0:
            self.print_model_summary()
        if params.plot_model > 0:
            self.plot_model_summary()

        if params.verbose > 1:
            self.model.summary()


class TRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim=32, warmup=250, learning_rate=0.001):
        super(TRSchedule, self).__init__()

        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)
        self.warmup = tf.cast(warmup, tf.float32)
        self.learning_rate = learning_rate * tf.math.sqrt(self.warmup)
        # Compensate for model dimension.
        self.learning_rate *= tf.math.sqrt(32.0/model_dim)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup ** -1.5)
        return self.learning_rate * tf.math.minimum(arg1, arg2)
