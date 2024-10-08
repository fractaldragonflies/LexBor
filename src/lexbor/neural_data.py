"""
Prepare data for use by neural modules.
Provide functions to access data.
"""
# TODO Review whether shuffle works correctly with how I've structured x and y.

# Import Python standard libraries
# from collections import Counter
# from pathlib import Path
import math
import random
import numpy as np

# Import 3rd-party libraries
import attr
import tensorflow as tf
import keras_nlp

# Build namespace
import lexbor.config as cfg

# output_path = Path(cfg.BaseSettings().output_path).resolve()
# logger = util.get_logger(__name__)


@attr.s
class NeuralTokenizer(keras_nlp.tokenizers.Tokenizer):
    """
    Provides tokens for language model.


    Parameters
    ----------
    vocab: Vocab
        Segment to id/ id to segment translation dictionary.
    prefix_start: bool, False
        Whether to prefix segments with start_id.
    postfix_stop: bool, False
        Whether to postfix segments with stop_id.
    """

    vocab = attr.ib(default=None)
    prefix_start = attr.ib(default=False)
    postfix_stop = attr.ib(default=False)

    def __attrs_pre_init__(self):
        super().__init__()

    @property
    def start_id(self):
        return self.vocab.start_id
    @property
    def stop_id(self):
        return self.vocab.stop_id
    @property
    def unk_id(self):
        return self.vocab.unk_id
    @property
    def nul_id(self):
        return self.vocab.nul_id

    def get_vocab(self):
        return self.vocab

    def vocab_size(self):
        return self.vocab.size

    def token_to_id(self, token):
        return self.vocab.get(token)

    def id_to_token(self, tkn_id):
        return self.vocab.get(tkn_id)

    def tokenize(self, inputs):
        # Word forms are already segmented except for start and stop symbols.
        # Inputs contain just the segmented IPA tokens.
        # Add start and stop if pre/post-fix enabled and convert to token ids.
        def tokenize_word(word):
            return (
                ([self.start_id] if self.prefix_start else [])
                + [self.vocab.get(x) for x in word]
                + ([self.stop_id] if self.postfix_stop else [])
                )
        if isinstance(inputs[0], list):
            return list(map(tokenize_word, inputs))
        else:
            return tokenize_word(inputs)

    def detokenize(self, inputs):
        # Inputs are contain token ids including start, stop, and padding.
        # Drop padding, start and stop, and convert to segments.
        def detokenize_word(word):
            # A word should have sequence start segments stop padding,
            # where start, stop and padding are not present within the word.
            # Word may extend to max size without stop or padding.

            start = 1 if word[0] == self.start_id else 0
            stop = (word.index(self.stop_id) if self.stop_id in word
                    else word.index(self.nul_id) if self.nul_id in word
                    else len(word)+1)
            return (
                [self.vocab.get(x) for x in word[start:stop]]
                )
        if isinstance(inputs[0], list):
            return list(map(detokenize_word, inputs))
        else:
            return detokenize_word(inputs)


@attr.s
class Vocab:
    """ Ignore option of using '+' as delimiter for now.
        data: [[str]]
            Each row is a list of segments.
    """
    data = attr.ib()
    null = attr.ib(default="<nul>")
    unk = attr.ib(default="<unk>")
    start = attr.ib(default="<s>")
    stop = attr.ib(default="</s>")
    nul_id = attr.ib(default=0)
    unk_id = attr.ib(default=1)
    start_id = attr.ib(default=2)
    stop_id = attr.ib(default=3)

    def __attrs_post_init__(self):
        self.vocab = {
            self.null: self.nul_id,
            self.unk: self.unk_id,
            self.start: self.start_id,
            self.stop: self.stop_id,
            self.nul_id: self.null,
            self.unk_id: self.unk,
            self.start_id: self.start,
            self.stop_id: self.stop,
        }
        counter = 4
        # Convert list of lists to list of unique tokens.
        for token in np.unique(np.concatenate(self.data).flat):
            self.vocab[token] = counter
            self.vocab[counter] = token
            counter += 1

        self.data = None  # Once used, no need to keep raw data.

    def __getitem__(self, item):
        return self.vocab[item]

    def get(self, item):
        if isinstance(item, str):
            return self.vocab.get(item, self.unk_id)
        else:
            return self.vocab.get(item, self.unk)

    def __len__(self):
        return len(self.vocab)

    @property
    def size(self):
        # Count only in 1 direction, not both forward and reverse translation.
        return len(self.vocab) // 2

# Problem: tensors must be rectangular. Ragged edges not allowed.
# This works with text as text is single string.
# When tokenized and packed it still ends up as single records with padding.
#

@attr.s
class PackageSequences:
    """
    Package segmented strings into id sequences for neural processing, and
    unpackage id sequences into segmented strings for human review.

    Constructed as a separate class for use by various modules.

    Parameters
    ----------
    vocab: {str:int} + {int:str}
        Dictionary and reverse diction of segmented strings and
        corresponding int ids.
    token_maxlen: int, default = 32
        Maximum length of int ids or str segments.
    """

    vocab = attr.ib(default=None)
    token_maxlen = attr.ib(default=32)
    tk = attr.ib(init=False)
    assert vocab is not None

    def __attrs_post_init__(self):
        self.tk = NeuralTokenizer(self.vocab)

    def get_tokenizer(self):
        return self.tk

    def process_segments(self, data, tokenized=False):
        """
            Design decisions for parallel x and y input and output data.
            With x and y offset by 1 for estimation of word entropy.
            Use multiple packers, configured for x as input and y as output.
            No need do offset, as use of packers with start or stop handles.

        data: [[str]] or [[int]]
            List of lists of segmented strings (segmented IPA), or
            list of lists of token_ids corresponding to segmented strings.
        tokenized: bool, default = False
            Whether segmented strings already tokenized into token_ids
        """
        # Tokenize if not already tokenized, and then only once.
        if tokenized: token_ids = data
        else: token_ids = self.tk(data)

        packer_x = keras_nlp.layers.StartEndPacker(
            sequence_length=self.token_maxlen,
            start_value=self.tk.start_id,
            end_value=None,
            pad_value=self.tk.nul_id,
            return_padding_mask=False
        )
        packer_y = keras_nlp.layers.StartEndPacker(
            sequence_length=self.token_maxlen,
            start_value=None,
            end_value=self.tk.stop_id,
            pad_value=self.tk.nul_id,
            return_padding_mask=False
        )

        token_ids_x = packer_x(token_ids)
        token_ids_y = packer_y(token_ids)
        return token_ids_x, token_ids_y

    # Option to add further methods, such as to deprocess segments.


@attr.s
class NeuralData:
    """
    Save train and test data in data store.
    Method to extract fit and val stores from train.
    Vocabulary calculated from train if not provided.
    Train, test, fit, val defined as tf.data.DataSet(s).

    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as indigenous or borrowed 0/1).
    test_data : [str, [str], int]
        Same structure as train.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary.
        Calculated from train if not provided.
    val_split: float, 0.0
        Proportion of training data to use for validation.
    batch_size: int, 32
        Batch size for data files.
    token_maxlen: int, 32
        Maximum length of token sequence.
    settings: NeuralSettings
        Default settings for various data, word model,
        and application variables.
    """

    train_data = attr.ib(default=None)
    test_data = attr.ib(default=None)
    vocab = attr.ib(default=None)
    val_split = attr.ib(default=None)
    batch_size = attr.ib(default=None)
    token_maxlen = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings())

    pkg = attr.ib(init=False)
    # train_len = attr.ib(init=False)
    # test_len = attr.ib(init=False)
    # fit_len = attr.ib(init=False)
    # val_len = attr.ib(init=False)

    fit_data = attr.ib(init=False, default=None)
    val_data = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self.vocab = self.vocab or Vocab([x[1] for x in self.train_data])
        self.val_split = (self.val_split if self.val_split is not None
                          else self.settings.val_split)
        self.batch_size = (self.batch_size if self.batch_size is not None
                           else self.settings.batch_size)
        self.token_maxlen = (self.token_maxlen if self.token_maxlen is not None
                             else self.settings.token_maxlen)
        self.pkg = PackageSequences(self.vocab, self.token_maxlen)
        self.__prepare_fit_val()  # Avoid extra method call.

    def __prepare_fit_val(self):

       if self.val_split == 0.0:
           self.fit_data = self.train_data
           self.val_data = None
       else:
           split = (int(self.val_split) if self.val_split >= 1
                    else math.ceil(len(self.train_data) * self.val_split))
           random.shuffle(self.train_data)
           self.val_data = self.train_data[:split]
           self.fit_data = self.train_data[split:]

    def get_dataset(self, data):
        """
            Use only segmented IPA (or similar) for resulting dataset.
            Would need ot override for dirct prediction with labels.
            Define and return dataset based on data.
            Return as None if no data.

        Parameters
        ----------
        data : [[str, [str], int]]
            Where initial str is an id,
            list of str are word character sequences (segmentd IPA),
            and final int is label for classification of word.
            Construct tf.data.DataSet corresponding to data.
        Returns
        -------
        dataset: inherited from tf.data.Dataset

        """
        if data is None or len(data) == 0:
            return None
        pkg_sequences = self.pkg.process_segments([row[1] for row in data])
        dataset = NeuralBatchDataset.from_tensor_slices(pkg_sequences)
        # dataset.length = len(pkg_sequences[0])
        # dataset.token_maxlen = self.token_maxlen
        dataset.batch_size = self.batch_size
        # Can append operations before use in train or test.
        # return (dataset
        #         .batch(self.batch_size)
        #         .shuffle(len(dataset))
        #         .prefetch(tf.data.AUTOTUNE))
        return dataset

    @property
    def train(self):
        if hasattr(self, "_train"):
            return(self._train)# pylint: disable:access-member-before-definition
        self._train = self.get_dataset(self.train_data)
        return self._train

    @property
    def fit(self):
        if hasattr(self, "_fit"):
            return self._fit  # pylint: disable:access-member-before-definition
        self._fit = self.get_dataset(self.fit_data)
        return self._fit

    @property
    def val(self):
        if hasattr(self, "_val"):
            return self._val  # pylint: disable:access-member-before-definition
        self._val = self.get_dataset(self.val_data)
        return self._val

    @property
    def test(self):
        if hasattr(self, "_test"):
            return self._test  # pylint: disable:access-member-before-definition
        self._test = self.get_dataset(self.test_data)
        return self._test

@attr.s
class NeuralBatchDataset(tf.data.Dataset):
    """
    Construct tf.data.Dataset and make available for neural processing.

    Parameters
    ----------
    pkg_sequences : [[int]], [[int]]
        Where character sequences are coded as int with appropriate,
        start, stop, and pad codes as well. Input sequences [0] are
        offset by 1 for output sequences [1] for use in training.
    token_maxlen: int
        Maximum sequence length.
    batch_size: int
        Batch size to be used for dataset.

    """
    # pkg_sequences = attr.ib(init=False, repr=False)
    # token_maxlen = attr.ib(init=False, default=None)
    batch_size = attr.ib(init=False, default=None)
    # length = attr.ib(init=False, default=None)

    # def __len__(self):
    #     return self.length

    # def get_token_datasets(self):

    #     train_tks = [x[1] for x in self.train]
    #     test_tks = [x[1] for x in self.test]

    #     # Convert ragged list of tokens into rectangular tensor of token ids
    #     train_tk_ids = self.pkg.process_segments(train_tks)
    #     self.train_len = len(train_tk_ids[0])  # From token_ids_x.
    #     train_prep = (tf.data.Dataset.from_tensor_slices(train_tk_ids)
    #                   .batch(self.batch_size).prefetch(tf.data.AUTOTUNE))

    #     test_tk_ids = self.pkg.process_segments(test_tks)
    #     self.test_len = len(test_tk_ids[0])
    #     test_prep = (tf.data.Dataset.from_tensor_slices(test_tk_ids)
    #                  .batch(self.batch_size).prefetch(tf.data.AUTOTUNE))

    #     return train_prep, test_prep

    # def get_fit_val_datasets(self, train_prep, val_split=None):
    #     """
    #     Shuffle train file and split into fit and val.
    #     This function could be invoked every multiple epochs
    #     to obtain new fit and val datasets during training.

    #     Parameters
    #     ----------
    #     train_prep : tf.data.Dataset
    #         Token sequences of segmented IPA for training.
    #     val_split: numeric
    #         Fraction or size value for validation dataset.

    #     Returns
    #     -------
    #     fit_prep : tf.data.Dataset
    #         Token sequences of segmented IPA for fit.
    #     val_prep : tf.data.Dataset
    #         Token sequences of segmented IPA for validation.

    #     """
    #     # train = train_prep.unbatch()
    #     train = train_prep.shuffle(self.train_len).unbatch()

    #     val_split = val_split if val_split is not None else self.val_split
    #     # print("val_split", val_split, "train_len", self.train_len)
    #     if val_split == 0.0:
    #         fit = train.batch(self.batch_size)  # .prefetch(tf.data.AUTOTUNE)
    #         self.fit_len = self.train_len
    #         # fit = fit.shuffle(self.fit_len, reshuffle_each_iteration=True)
    #         val = None
    #     else:
    #         split = (int(val_split) if val_split >= 1
    #                  else math.ceil(self.train_len * val_split))
    #         val = train.take(split).batch(self.batch_size)  # .prefetch(tf.data.AUTOTUNE)
    #         fit = train.skip(split).batch(self.batch_size)  # .prefetch(tf.data.AUTOTUNE)
    #         self.val_len = split
    #         self.fit_len = self.train_len - split
    #         # fit = fit.shuffle(self.fit_len, reshuffle_each_iteration=True)
    #         # val = val.shuffle(self.val_len, reshuffle_each_iteration=True)
    #     return fit, val

@attr.s
class NeuralDataGen:
    """
    Save train and test data in data store.
    Method to extract fit and val stores from train.
    Vocabulary calculated from train if not provided.
    Train, test, fit, val  available as generators.

    Parameters
    ----------
    train_data : [str, [str], int]
        Training data as rows of words. Each word consists of an identifier,
        token represented as list of character segments, and loan status
        (typically as indigenous or borrowed 0/1).
    test_data : [str, [str], int]
        Same structure as train.
    vocab: {[str] : int}, optional
        Symbol to id translation dictionary.
        Calculated from train if not provided.
    val_split: float, 0.0
        Proportion of training data to use for validation.
    batch_size: int, 32
        Batch size for data files.
    token_maxlen: int, 32
        Maximum length of token sequence.
    settings: [key: value] of default settings.
    """

    train_data = attr.ib(default=None)
    test_data = attr.ib(default=None)
    vocab = attr.ib(default=None)
    val_split = attr.ib(default=None)
    batch_size = attr.ib(default=None)
    token_maxlen = attr.ib(default=None)
    settings = attr.ib(default=cfg.NeuralSettings())

    pkg = attr.ib(init=False)
    # tk = attr.ib(init=False)

    fit_data = attr.ib(init=False, default=None)
    val_data = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self.vocab = self.vocab or Vocab([x[1] for x in self.train_data])
        self.val_split = (self.val_split if self.val_split is not None
                          else self.settings.val_split)
        self.batch_size = (self.batch_size if self.batch_size is None
                           else self.settings.batch_size)
        self.token_maxlen = (self.token_maxlen if self.token_maxlen is not None
                             else self.settings.token_maxlen)
        self.pkg = PackageSequences(self.vocab, self.token_maxlen)

        self.__prepare_fit_val()  # Avoid extra method call.

    def __prepare_fit_val(self):

        if self.val_split == 0.0:
            self.fit_data = self.train_data
            self.val_data = None
        else:
            split = (int(self.val_split) if self.val_split >= 1
                     else math.ceil(len(self.train_data) * self.val_split))
            random.shuffle(self.train_data)
            self.val_data = self.train_data[:split]
            self.fit_data = self.train_data[split:]


    def get_generator(self, data):
        """
            Only segmented IPA used for data.
            Would need to override for direct prediction.
            Return generator as None if no data.
        """
        if data is None or len(data) == 0:
            return None
        tk = self.pkg.get_tokenizer()
        token_ids = tk([row[1] for row in data])
        return KerasBatchGenerator(
            token_ids,
            # token_maxlen = self.token_maxlen,
            batch_size=self.batch_size,
            pkg = self.pkg,
            settings=self.settings,
        )

    @property
    def train(self):
        if hasattr(self, "_train"):
            return self._train  # pylint: disable:access-member-before-definition
        self._train = self.get_generator(self.train_data)
        return self._train

    @property
    def fit(self):
        if hasattr(self, "_fit"):
            return self._fit  # pylint: disable:access-member-before-definition
        self._fit = self.get_generator(self.fit_data)
        return self._fit

    @property
    def val(self):
        if hasattr(self, "_val"):
            return self._val  # pylint: disable:access-member-before-definition
        self._val = self.get_generator(self.val_data)
        return self._val

    @property
    def test(self):
        if hasattr(self, "_test"):
            return self._test  # pylint: disable:access-member-before-definition
        self._test = self.get_generator(self.test_data)
        return self._test

@attr.s
class KerasBatchGenerator:
    """
    Construct data generators for neural network training and test.

    Parameters
    ----------
    token_ids : [[int]]
        List of token_ids corresponding to segmented IPA.
    token_maxlen: int, optional
        Maximum length of token sequence.
        Defaults to configuration token maxlen.
    batch_size: int, optional
        Batch size to use in neural network fit.
        Defaults to configuration batch size.
    pkg: PackageSequences
        Packager already instantiated for tokenizing and packing data.
    """

    token_ids = attr.ib(repr=False)
    # token_maxlen = attr.ib(default=None)
    batch_size = attr.ib(default=None)
    pkg = attr.ib(default=None, repr=False)
    settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

    def __attrs_post_init__(self):
        # self.token_maxlen = (self.token_maxlen if self.token_maxlen is not None
        #                      else self.settings.token_maxlen)
        self.batch_size = (self.batch_size if self.batch_size is not None
                           else self.settings.batch_size)

    def __len__(self):
        return len(self.token_ids)

    def generate(self):
        """
            Design decisions for parallel x and y input and output data.
            With x and y offset by 1 for estimation of word entropy.
            Use multiple packers, configured for x as input and y as output.
            Packers with start or stop handles x and y offset.
        """

        while True:
            # Shuffle each dataset at each epoch.
            # Unable to reallocate fit and val from train data.
            random.shuffle(self.token_ids)

# Python generator still works in tf 2+. Consider PyDataset for enhancement.
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# This example uses return instead of yield to provide data batches.
# Also: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
# Python generator may also still work: https://towardsdatascience.com/writing-custom-keras-generators-fe815d992c5a
            # Last batch may be partial batch.
            # Round up on batches to not miss any data.
            count = 0
            while count < len(self.token_ids):
                next_count = count + self.batch_size
                if next_count <= len(self.token_ids):
                    x = self.token_ids[count:next_count]
                    count += self.batch_size
                else:  # Use partial batch.
                    x = self.token_ids[count:]
                    count = len(self.token_ids)
                x, y = self.pkg.process_segments(x, tokenized=True)

                yield x, y

# @attr.s
# class NeuralData:
#     """
#     Save training and testing data in data store. Provide data generators and alphabet.
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

#     training = attr.ib(default=[])
#     testing = attr.ib(default=[])
#     vocab = attr.ib(default=None)
#     val_split = attr.ib(default=None)
#     settings = attr.ib(default=cfg.NeuralSettings())

#     def __attrs_post_init__(self):
#         self.all_data = self.training + self.testing
#         random.shuffle(self.all_data)
#         self.vocab = self.vocab or Vocab([x[1] for x in self.all_data])
#         self.vocab.dlm_is_plus = self.settings.dlm_is_plus

#         random.shuffle(self.testing)
#         self.val_split = (
#             self.settings.val_split if self.val_split is None else self.val_split
#         )
#         self.fit, self.val = util.train_test_split(self.training, self.val_split)
#         self.label_counts = dict(Counter([x[2] for x in self.training]))

#         if self.settings.verbose:
#             logger.info(
#                 f"Train length: {len(self.training)}"
#                 + f", fit length: {len(self.fit)}"
#                 + f", val length: {len(self.val) if self.val else 0}"
#                 + f", test length: {len(self.testing) if self.testing else 0}."
#             )

#     def translate(self, sequences):
#         """
#         Translate words to the internal numeric alphabet.
#         """
#         return [self.vocab.translate(t) for t in sequences]

#     def get_data_tokens(self, data):
#         return [x[1] for x in data]

#     def get_data_tokens_ids(self, data):
#         return self.translate([x[1] for x in data])

#     def get_tokens_ids(self, sequences):
#         return self.translate(sequences)

#     def get_data_states(self, data):
#         return [x[2] for x in data]

#     # Only sound segments used for data.
#     # Would need to override for direct prediction.
#     def get_batcher(self, data):
#         # Return generator as None if no data.
#         if data is None or len(data) == 0:
#             return None
#         return KerasBatchGenerator(
#             self.translate([x[1] for x in data]),
#             batch_size=self.settings.batch_size,
#             vocab_size=self.vocab.size,
#             settings=self.settings,
#         )

#     @property
#     def trainer(self):
#         if hasattr(self, "_trainer"):
#             return self._trainer  # pylint: disable:access-member-before-definition
#         self._trainer = self.get_batcher(self.fit)
#         return self._trainer

#     @property
#     def validator(self):
#         if hasattr(self, "_validator"):
#             return self._validator  # pylint: disable:access-member-before-definition
#         self._validator = self.get_batcher(self.val)
#         return self._validator

#     @property
#     def tester(self):
#         if hasattr(self, "_tester"):
#             return self._tester  # pylint: disable:access-member-before-definition
#         self._tester = self.get_batcher(self.testing)
#         return self._tester


# @attr.s
# class KerasBatchGenerator:
#     """
#     Construct data generators for neural network training and test.
#     """
#     """
#     Construct a generator for the neural network

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

#     data = attr.ib(repr=False)
#     batch_size = attr.ib(default=None)
#     vocab_size = attr.ib(default=None)
#     skip_step = attr.ib(default=None)
#     settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

#     def __attrs_post_init__(self):
#         self.batch_size = self.batch_size or self.settings.batch_size
#         self.skip_step = self.skip_step or self.settings.skip_step
#         self.current_idx = 0
#         self.data_len = len(self.data)
#         # Test for unknowns.

#     def generate(self, sample=None):
#         # Randomize order of words.
#         data = list(self.data)
#         random.shuffle(data)
#         count = 0
#         while not sample or count < sample:
#             count += 1
#             x_lst = []
#             y_lst = []
#             for _ in range(self.batch_size):
#                 if self.current_idx >= self.data_len:
#                     self.current_idx = 0
#                 # Build 2-D list of lists of ids for each word.
#                 # Apply Keras pad_sequences (post padding).
#                 # Yield x and y numpy arrays.
#                 # Truncate last symbol because we don't have more y to predict.
#                 x_lst.append(data[self.current_idx][:-1])

#                 # Treat y as sparse.
#                 y_lst.append(data[self.current_idx][1:])

#                 self.current_idx += 1

#             self.current_idx += random.randint(0, self.skip_step)
#             # Truncate word lengths if needed.
#             maxlen = max([len(token_ids) for token_ids in x_lst])
#             if maxlen > self.settings.token_maxlen + 1:
#                 maxlen = self.settings.token_maxlen + 1

#             x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
#             y = pad_sequences(y_lst, padding="post", maxlen=maxlen)
#             yield x, y


# @attr.s
# class NeuralDataDirect(NeuralData):
#     """
#     Prepare training and test data for direct method models.
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

#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()

#     # Override for direct prediction.
#     def get_batcher(self, data):
#         # Return generator as None if no data.
#         if data is None or len(data) == 0:
#             return None
#         data_x_y = list(zip(self.translate([x[1] for x in data]), self.get_data_states(data)))
#         return KerasBatchGeneratorDirect(
#             data_x_y,  # Tokens and status go together
#             batch_size=self.settings.batch_size,
#             vocab_size=self.vocab.size,
#             settings=self.settings,
#         )


# @attr.s
# class KerasBatchGeneratorDirect:
#     """
#     Make generators for direct method models.
#     """
#     """
#     Construct a generator for the neural network

#     Parameters
#     ----------
#     data : [[int] int]
#         list of tokens and states. Tokens are represented by lists of ids.
#         Each list of ids corresponds to a list of character string segments.
#         States are represented as state ids.
#     batch_size : int, optional
#         Batch size to use in neural network fit.
#         If not given, the configuration batch size is used.
#     vocab_size : int
#         Size of the vocabulary - number of string keys in the vocabulary.
#     skip_step : int, optional
#         Maximum leap in data sequence. Step chosen at random up to maximum.
#     """

#     data = attr.ib(repr=False)
#     batch_size = attr.ib(default=None)
#     vocab_size = attr.ib(default=None)
#     skip_step = attr.ib(default=None)
#     settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

#     def __attrs_post_init__(self):
#         self.batch_size = self.batch_size or self.settings.batch_size
#         self.skip_step = self.skip_step or self.settings.skip_step
#         self.current_idx = 0
#         self.data_len = len(self.data)
#         # Test for unknowns.

#     def generate(self, sample=None):
#         # Randomize order of words.
#         data = list(self.data)
#         random.shuffle(data)
#         count = 0
#         while not sample or count < sample:
#             count += 1
#             x_lst = []
#             y_lst = []
#             for _ in range(self.batch_size):
#                 if self.current_idx >= self.data_len:
#                     self.current_idx = 0
#                 # Build list of lists of ids over words.
#                 # Build list of states over words.
#                 # Apply Keras pad_sequences (post padding).
#                 # Yield x and y -- sparse numpy arrays.
#                 x_lst.append(data[self.current_idx][0])
#                 y_lst.append(data[self.current_idx][1])

#                 self.current_idx += 1

#             self.current_idx += random.randint(0, self.skip_step)
#             # Truncate word lengths if needed.
#             # Both start and stop codings included.
#             maxlen = max([len(token_ids) for token_ids in x_lst])
#             if maxlen > self.settings.token_maxlen + 2:
#                 maxlen = self.settings.token_maxlen + 2

#             x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
#             # y = tf.convert_to_tensor(y_lst, dtype=tf.int32)
#             y = np.array(y_lst, dtype=int)
#             yield x, y


# @attr.s
# class NeuralDataDirectHot(NeuralData):
#     """
#     Data in 1-hot format as required to use the F1Score measures from add-on library.
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

#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()

#     # Override for direct prediction.
#     def get_batcher(self, data):
#         # Return generator as None if no data.
#         if data is None or len(data) == 0:
#             return None
#         data_x_y = list(zip(self.translate([x[1] for x in data]), self.get_data_states(data)))
#         return KerasBatchGeneratorDirectHot(
#             data_x_y,  # Tokens and status go together
#             batch_size=self.settings.batch_size,
#             vocab_size=self.vocab.size,
#             label_size=len(self.label_counts),
#             settings=self.settings,
#         )


# @attr.s
# class KerasBatchGeneratorDirectHot:
#     """
#     Data generators for direct methods with 1-hot encoding.
#     """
#     """
#     Construct a generator for the neural network

#     Parameters
#     ----------
#     data : [[int] int]
#         list of tokens and states. Tokens are represented by lists of ids.
#         Each list of ids corresponds to a list of character string segments.
#         States are represented as state ids.
#     batch_size : int, optional
#         Batch size to use in neural network fit.
#         If not given, the configuration batch size is used.
#     vocab_size : int
#         Size of the vocabulary - number of string keys in the vocabulary.
#     label_size : int
#         Size of the output labels - number of distinct labels.
#     skip_step : int, optional
#         Maximum leap in data sequence. Step chosen at random up to maximum.
#     """

#     data = attr.ib(repr=False)
#     batch_size = attr.ib(default=None)
#     vocab_size = attr.ib(default=None)
#     skip_step = attr.ib(default=None)
#     label_size = attr.ib(default=None)
#     settings = attr.ib(default=cfg.NeuralSettings(), repr=False)

#     def __attrs_post_init__(self):
#         self.batch_size = self.batch_size or self.settings.batch_size
#         self.skip_step = self.skip_step or self.settings.skip_step
#         self.current_idx = 0
#         self.data_len = len(self.data)
#         # Test for unknowns.

#     def generate(self, sample=None):
#         # Randomize order of words.
#         data = list(self.data)
#         random.shuffle(data)
#         count = 0
#         while not sample or count < sample:
#             count += 1
#             x_lst = []
#             y_lst = []
#             for _ in range(self.batch_size):
#                 if self.current_idx >= self.data_len:
#                     self.current_idx = 0
#                 # Build list of lists of ids over words.
#                 # Build list of states over words.
#                 # Apply Keras pad_sequences (post padding).
#                 # Yield x and y -- sparse numpy arrays.
#                 x_lst.append(data[self.current_idx][0])
#                 # y_lst.append(data[self.current_idx][1])
#                 temp_y = data[self.current_idx][1]
#                 y_lst.append(to_categorical(temp_y, num_classes=self.label_size))

#                 self.current_idx += 1

#             self.current_idx += random.randint(0, self.skip_step)
#             # Truncate word lengths if needed.
#             # Both start and stop codings included.
#             maxlen = max([len(token_ids) for token_ids in x_lst])
#             if maxlen > self.settings.token_maxlen + 2:
#                 maxlen = self.settings.token_maxlen + 2

#             x = pad_sequences(x_lst, padding="post", maxlen=maxlen)
#             y = pad_sequences(y_lst, padding="post")
#             yield x, y


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

