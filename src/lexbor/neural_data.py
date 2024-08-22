# Copied from Pybor source
# Import Python standard libraries
from collections import Counter
from pathlib import Path

import random
import numpy as np

# Import 3rd-party libraries
import attr

# import tensorflow as tf
from keras.utils import pad_sequences
from keras.utils import to_categorical

# Build namespace
import pybor2.util as util
import pybor2.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()

logger = util.get_logger(__name__)

START_VAL = 2
END_VAL = 3
PLUS_VAL = 4  # In some coding + is gap between words, and begin and end of words.


@attr.s
class Vocab:
    data = attr.ib()
    dlm_is_plus = attr.ib(default=False)
    null = attr.ib(default="<nul>")
    start = attr.ib(default="<s>")
    end = attr.ib(default="</s>")
    unknown = attr.ib(default="<unk>")
    plus = attr.ib(default="+")

    def __attrs_post_init__(self):
        counter = 4
        self.vocab = {
            self.null: 0,
            self.unknown: 1,
            self.start: START_VAL,
            self.end: END_VAL,
            self.plus: PLUS_VAL,
            0: self.null,
            1: self.unknown,
            START_VAL: self.start,
            END_VAL: self.end,
            PLUS_VAL: self.plus,
        }
        for tokens in self.data:
            for token in tokens:
                try:
                    self.vocab[token]
                except KeyError:
                    self.vocab[token] = counter
                    self.vocab[counter] = token
                    counter += 1

        self.data = None  # Once used, no need to keep raw data.

    def __getitem__(self, item):
        return self.vocab[item]

    def get(self, item, alternative=None):
        return self.vocab.get(item, alternative)

    def __len__(self):
        return len(self.vocab)

    # Unknown value is 1 and used here.
    def translate(self, word):
        if self.dlm_is_plus:
            return (
                [self.vocab[self.plus]]
                + [self.vocab.get(x, 1) for x in word]
                + [self.vocab[self.plus]]
                )
        else:
            return (
                [self.vocab[self.start]]
                + [self.vocab.get(x, 1) for x in word]
                + [self.vocab[self.end]]
                )

    @property
    def size(self):
        # Count only in 1 direction, not both forward and reverse translation.
        return len(self.vocab) // 2


@attr.s
class NeuralData:
    """
    Save training and testing data in data store. Provide data generators and alphabet.
    """
    """
    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as native or loan 0/1).
    test_data : [str, [str], int]
        Same as train_data.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary. Calculated from tokens if not provided.
        If multiple data stores are defined then a common vocab should be used.
    val_split: float, optional
        Proportion of training data to use for validation.
        Uses value from neural_cfg.data if not overridden here.
    """

    training = attr.ib(default=[])
    testing = attr.ib(default=[])
    vocab = attr.ib(default=None)
    val_split = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings())

    def __attrs_post_init__(self):
        self.all_data = self.training + self.testing
        random.shuffle(self.all_data)
        self.vocab = self.vocab or Vocab([x[1] for x in self.all_data])
        self.vocab.dlm_is_plus = self.settings.dlm_is_plus

        random.shuffle(self.testing)
        self.val_split = (
            self.settings.val_split if self.val_split is None else self.val_split
        )
        self.fit, self.val = util.train_test_split(self.training, self.val_split)
        self.label_counts = dict(Counter([x[2] for x in self.training]))

        if self.settings.verbose:
            logger.info(
                f"Train length: {len(self.training)}"
                + f", fit length: {len(self.fit)}"
                + f", val length: {len(self.val) if self.val else 0}"
                + f", test length: {len(self.testing) if self.testing else 0}."
            )

    def translate(self, sequences):
        """
        Translate words to the internal numeric alphabet.
        """
        return [self.vocab.translate(t) for t in sequences]

    def get_data_tokens(self, data):
        return [x[1] for x in data]

    def get_data_tokens_ids(self, data):
        return self.translate([x[1] for x in data])

    def get_tokens_ids(self, sequences):
        return self.translate(sequences)

    def get_data_states(self, data):
        return [x[2] for x in data]

    # Only sound segments used for data.
    # Would need to override for direct prediction.
    def get_batcher(self, data):
        # Return generator as None if no data.
        if data is None or len(data) == 0:
            return None
        return KerasBatchGenerator(
            self.translate([x[1] for x in data]),
            batch_size=self.settings.batch_size,
            vocab_size=self.vocab.size,
            settings=self.settings,
        )

    @property
    def trainer(self):
        if hasattr(self, "_trainer"):
            return self._trainer  # pylint: disable:access-member-before-definition
        self._trainer = self.get_batcher(self.fit)
        return self._trainer

    @property
    def validator(self):
        if hasattr(self, "_validator"):
            return self._validator  # pylint: disable:access-member-before-definition
        self._validator = self.get_batcher(self.val)
        return self._validator

    @property
    def tester(self):
        if hasattr(self, "_tester"):
            return self._tester  # pylint: disable:access-member-before-definition
        self._tester = self.get_batcher(self.testing)
        return self._tester


@attr.s
class KerasBatchGenerator:
    """
    Construct data generators for neural network training and test.
    """
    """
    Construct a generator for the neural network

    Parameters
    ----------
    data : [[int]]
        list of tokens. Tokens are represented by lists of ids.
        Each list of ids corresponds to a list of character string segments.
    batch_size : int, optional
        Batch size to use in neural network fit.
        If not given, the configuration batch size is used.
    vocab_size : int
        Size of the vocabulary - number of string keys in the vocabulary.
    skip_step : int, optional
        Maximum leap in data sequence. Step chosen at random up to maximum.
    """

    data = attr.ib(repr=False)
    batch_size = attr.ib(default=None)
    vocab_size = attr.ib(default=None)
    skip_step = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

    def __attrs_post_init__(self):
        self.batch_size = self.batch_size or self.settings.batch_size
        self.skip_step = self.skip_step or self.settings.skip_step
        self.current_idx = 0
        self.data_len = len(self.data)
        # Test for unknowns.

    def generate(self, sample=None):
        # Randomize order of words.
        data = list(self.data)
        random.shuffle(data)
        count = 0
        while not sample or count < sample:
            count += 1
            x_lst = []
            y_lst = []
            for _ in range(self.batch_size):
                if self.current_idx >= self.data_len:
                    self.current_idx = 0
                # Build 2-D list of lists of ids for each word.
                # Apply Keras pad_sequences (post padding).
                # Yield x and y numpy arrays.
                # Truncate last symbol because we don't have more y to predict.
                x_lst.append(data[self.current_idx][:-1])

                # Treat y as sparse.
                y_lst.append(data[self.current_idx][1:])

                self.current_idx += 1

            self.current_idx += random.randint(0, self.skip_step)
            # Truncate word lengths if needed.
            maxlen = max([len(token_ids) for token_ids in x_lst])
            if maxlen > self.settings.token_maxlen + 1:
                maxlen = self.settings.token_maxlen + 1

            x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
            y = pad_sequences(y_lst, padding="post", maxlen=maxlen)
            yield x, y


@attr.s
class NeuralDataDirect(NeuralData):
    """
    Prepare training and test data for direct method models.
    """
    """
    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as native or loan 0/1).
    test_data : [str, [str], int]
        Same as train_data.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary. Calculated from tokens if not provided.
        If multiple data stores are defined then a common vocab should be used.
    val_split: float, optional
        Proportion of training data to use for validation.
        Uses value from neural_cfg.data if not overridden here.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    # Override for direct prediction.
    def get_batcher(self, data):
        # Return generator as None if no data.
        if data is None or len(data) == 0:
            return None
        data_x_y = list(zip(self.translate([x[1] for x in data]), self.get_data_states(data)))
        return KerasBatchGeneratorDirect(
            data_x_y,  # Tokens and status go together
            batch_size=self.settings.batch_size,
            vocab_size=self.vocab.size,
            settings=self.settings,
        )


@attr.s
class KerasBatchGeneratorDirect:
    """
    Make generators for direct method models.
    """
    """
    Construct a generator for the neural network

    Parameters
    ----------
    data : [[int] int]
        list of tokens and states. Tokens are represented by lists of ids.
        Each list of ids corresponds to a list of character string segments.
        States are represented as state ids.
    batch_size : int, optional
        Batch size to use in neural network fit.
        If not given, the configuration batch size is used.
    vocab_size : int
        Size of the vocabulary - number of string keys in the vocabulary.
    skip_step : int, optional
        Maximum leap in data sequence. Step chosen at random up to maximum.
    """

    data = attr.ib(repr=False)
    batch_size = attr.ib(default=None)
    vocab_size = attr.ib(default=None)
    skip_step = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

    def __attrs_post_init__(self):
        self.batch_size = self.batch_size or self.settings.batch_size
        self.skip_step = self.skip_step or self.settings.skip_step
        self.current_idx = 0
        self.data_len = len(self.data)
        # Test for unknowns.

    def generate(self, sample=None):
        # Randomize order of words.
        data = list(self.data)
        random.shuffle(data)
        count = 0
        while not sample or count < sample:
            count += 1
            x_lst = []
            y_lst = []
            for _ in range(self.batch_size):
                if self.current_idx >= self.data_len:
                    self.current_idx = 0
                # Build list of lists of ids over words.
                # Build list of states over words.
                # Apply Keras pad_sequences (post padding).
                # Yield x and y -- sparse numpy arrays.
                x_lst.append(data[self.current_idx][0])
                y_lst.append(data[self.current_idx][1])

                self.current_idx += 1

            self.current_idx += random.randint(0, self.skip_step)
            # Truncate word lengths if needed.
            # Both start and stop codings included.
            maxlen = max([len(token_ids) for token_ids in x_lst])
            if maxlen > self.settings.token_maxlen + 2:
                maxlen = self.settings.token_maxlen + 2

            x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
            # y = tf.convert_to_tensor(y_lst, dtype=tf.int32)
            y = np.array(y_lst, dtype=int)
            yield x, y


@attr.s
class NeuralDataDirectHot(NeuralData):
    """
    Data in 1-hot format as required to use the F1Score measures from add-on library.
    """
    """
    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as native or loan 0/1).
    test_data : [str, [str], int]
        Same as train_data.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary. Calculated from tokens if not provided.
        If multiple data stores are defined then a common vocab should be used.
    val_split: float, optional
        Proportion of training data to use for validation.
        Uses value from neural_cfg.data if not overridden here.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    # Override for direct prediction.
    def get_batcher(self, data):
        # Return generator as None if no data.
        if data is None or len(data) == 0:
            return None
        data_x_y = list(zip(self.translate([x[1] for x in data]), self.get_data_states(data)))
        return KerasBatchGeneratorDirectHot(
            data_x_y,  # Tokens and status go together
            batch_size=self.settings.batch_size,
            vocab_size=self.vocab.size,
            label_size=len(self.label_counts),
            settings=self.settings,
        )


@attr.s
class KerasBatchGeneratorDirectHot:
    """
    Data generators for direct methods with 1-hot encoding.
    """
    """
    Construct a generator for the neural network

    Parameters
    ----------
    data : [[int] int]
        list of tokens and states. Tokens are represented by lists of ids.
        Each list of ids corresponds to a list of character string segments.
        States are represented as state ids.
    batch_size : int, optional
        Batch size to use in neural network fit.
        If not given, the configuration batch size is used.
    vocab_size : int
        Size of the vocabulary - number of string keys in the vocabulary.
    label_size : int
        Size of the output labels - number of distinct labels.
    skip_step : int, optional
        Maximum leap in data sequence. Step chosen at random up to maximum.
    """

    data = attr.ib(repr=False)
    batch_size = attr.ib(default=None)
    vocab_size = attr.ib(default=None)
    skip_step = attr.ib(default=None)
    label_size = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

    def __attrs_post_init__(self):
        self.batch_size = self.batch_size or self.settings.batch_size
        self.skip_step = self.skip_step or self.settings.skip_step
        self.current_idx = 0
        self.data_len = len(self.data)
        # Test for unknowns.

    def generate(self, sample=None):
        # Randomize order of words.
        data = list(self.data)
        random.shuffle(data)
        count = 0
        while not sample or count < sample:
            count += 1
            x_lst = []
            y_lst = []
            for _ in range(self.batch_size):
                if self.current_idx >= self.data_len:
                    self.current_idx = 0
                # Build list of lists of ids over words.
                # Build list of states over words.
                # Apply Keras pad_sequences (post padding).
                # Yield x and y -- sparse numpy arrays.
                x_lst.append(data[self.current_idx][0])
                # y_lst.append(data[self.current_idx][1])
                temp_y = data[self.current_idx][1]
                y_lst.append(to_categorical(temp_y, num_classes=self.label_size))

                self.current_idx += 1

            self.current_idx += random.randint(0, self.skip_step)
            # Truncate word lengths if needed.
            # Both start and stop codings included.
            maxlen = max([len(token_ids) for token_ids in x_lst])
            if maxlen > self.settings.token_maxlen + 2:
                maxlen = self.settings.token_maxlen + 2

            x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
            y = pad_sequences(y_lst, padding="post")
            yield x, y


# @attr.s
# class NeuralDataForBack(NeuralData):
#     """
#     Prepare training and test data for forward-backward models.
#     """
#     """
#     Parameters
#     ----------
#     train_data : [str, [str], int]
#         Training data as rows of words. Each word consists of an identifier,
#         token represented as list of character segments, and loan status
#         (typically as native or loan 0/1).
#     test_data : [str, [str], int]
#         Same as train_data.
#     vocab: {[str] : int}, optional
#         Symbol to id translation dictionary. Calculated from tokens if not provided.
#         If multiple data stores are defined then a common vocab should be used.
#     val_split: float, optional
#         Proportion of training data to use for validation.
#         Uses value from neural_cfg.data if not overridden here.
#     """
#
#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()
#
#         # Only sound segments used for data.
#         # Would need to override for direct prediction.
#
#     def get_batcher(self, data):
#         # Return generator as None if no data.
#         if data is None or len(data) == 0:
#             return None
#         return KerasBatchGeneratorForBack(
#             self.translate([x[1] for x in data]),
#             batch_size=self.settings.batch_size,
#             vocab_size=self.vocab.size,
#             settings=self.settings,
#         )
#
#
# @attr.s
# class KerasBatchGeneratorForBack:
#     """
#     Construct data generators for neural network training and test.
#     """
#     """
#     Construct a generator for the neural network
#
#     Parameters
#     ----------
#     data : [[int]]
#         list of tokens. Tokens are represented by lists of ids.
#         Each list of ids corresponds to a list of character string segments.
#     batch_size : int, optional
#         Batch size to use in neural network fit.
#         If not given, the configuration batch size is used.
#     vocab_size : int
#         Size of the vocabulary - number of string keys in the vocabulary.
#     skip_step : int, optional
#         Maximum leap in data sequence. Step chosen at random up to maximum.
#     """
#
#     data = attr.ib(repr=False)
#     batch_size = attr.ib(default=None)
#     vocab_size = attr.ib(default=None)
#     skip_step = attr.ib(default=None)
#     settings = attr.ib(default=cfg.NeuralSettings(), repr=False)
#
#     def __attrs_post_init__(self):
#         self.batch_size = self.batch_size or self.settings.batch_size
#         self.skip_step = self.skip_step or self.settings.skip_step
#         self.current_idx = 0
#         self.data_len = len(self.data)
#         # Test for unknowns.
#
#     def generate(self, sample=None):
#         # Randomize order of words.
#         data = list(self.data)
#         random.shuffle(data)
#         count = 0
#         while not sample or count < sample:
#             count += 1
#             x_lst = []
#             x_back_lst = []
#             y_lst= []
#             y_back_lst = []
#             for _ in range(self.batch_size):
#                 if self.current_idx >= self.data_len:
#                     self.current_idx = 0
#                 # Build 2-D list of lists of ids for each word.
#                 # Apply Keras pad_sequences (post padding).
#                 # Yield x and y numpy arrays.
#                 # Truncate last symbol because we don't have more y to predict.
#                 x_lst.append(data[self.current_idx][:-1])
#                 x_back_lst.append(data[self.current_idx][:0:-1])
#
#                 # Treat y as sparse.
#                 y_lst.append(data[self.current_idx][1:])
#                 y_back_lst.append(data[self.current_idx][-2::-1])
#
#                 self.current_idx += 1
#
#             self.current_idx += random.randint(0, self.skip_step)
#             # Truncate word lengths if needed.
#             maxlen = max([len(token_ids) for token_ids in x_lst])
#             if maxlen > self.settings.token_maxlen + 1:
#                 maxlen = self.settings.token_maxlen + 1
#
#             x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
#             x_back = pad_sequences(x_back_lst, padding="post", maxlen=maxlen)
#             y = pad_sequences(y_lst, padding="post", maxlen=maxlen)
#             y_back = pad_sequences(y_back_lst, padding="post", maxlen=maxlen)
#
#             yield [x, x_back], [y, y_back]
