# -*- coding: utf-8 -*-
"""
Created on Mon Nov 9 13:30 2020

@author: johnmiller

Test neural_data.py

All test functions begin with 'test_'
Import testing, training, testing1, training1 from pybor.dev.data

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_neural_data.py --cov=lexbor.neural_data

Or to save as html report:
$ pytest tests/test_neural_data.py -—cov=lexbor.neural_data --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_neural_data.py --cov=lexbor.neural_data --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_neural_data.py::test_datasets --cov=lexbor.neural_data

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests, or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

Add the -s argument to show output from print statements.

"""
import math

import tensorflow as tf
import lexbor.config as cfg
from lexbor import util, wold
from lexbor.neural_data import NeuralTokenizer, Vocab
import lexbor.neural_data as nd

import pytest


def get_data_for_language(language=None, form='Tokens', test_split=0.2):
    wold_db = wold.get_wold_access()
    language_ = wold.check_wold_languages(wold_db, language)[0]

    table = wold_db.get_table(
        language=language_, form=form, classification='Borrowed')
    train, test = util.train_test_split(table, split=test_split)
    return train, test

settings = cfg.NeuralSettings()

# =============================================================================
# Test functions -- all begin with the string test_
#
# =============================================================================
def test_vocab():
    print('test_vocab')
    train, test = get_data_for_language(language='English', form='Tokens')
    all_data = train + test

    vocab = Vocab([x[1] for x in all_data])
    assert(vocab.size == 51)
    assert(len(vocab) == 102)
    # print(vocab.vocab)
    tok_id = vocab['ɔɪ']
    assert(vocab['ɔɪ'] == tok_id)
    assert(vocab.get('ɔɪ') == tok_id)
    assert(vocab.get('ɪɔ') == 1)
    assert(vocab.get(tok_id) == 'ɔɪ')
    assert(vocab.get(255) == '<unk>')

    assert(vocab.get(1) == '<unk>')
    assert(vocab.get(2) == '<s>')
    assert(vocab.get(3) == '</s>')
    assert(vocab.get('<nul>') == 0)
    assert(vocab.get(0) == '<nul>')

def test_tokenizer():
    print('test_tokenizer')
    train, test = get_data_for_language(language='English', form='Tokens')
    all_data = train + test
    vocab = Vocab([x[1] for x in all_data])

    tk = NeuralTokenizer(vocab,prefix_start=True, postfix_stop=True)
    vocab = tk.get_vocab()
    assert(vocab)
    assert(tk.vocab_size() == 51)
    assert(tk.start_id == 2)
    assert(tk.stop_id == 3)
    assert(tk.unk_id == 1)
    tok_id = vocab.get('uː')
    assert(tk.token_to_id('uː') == tok_id)
    assert(tk.id_to_token(tok_id) == 'uː')

    x = [x[1] for x in train[:2]]
    print(x)
    x_ = tk.tokenize(x)
    print(x_)
    assert(len(x_[0]) == len(x[0])+2)
    assert(x_[0][0] == tk.start_id)
    assert(x_[0][-1] == tk.stop_id)
    assert(x_[0][1] == vocab.get(x[0][0]))

    # x[1] from row 1 of train.
    x1 = train[1][1]
    print(x1)
    x1_ = tk.tokenize(x1)
    print(x1_)
    assert(x[1] == x1)
    assert(x_[1] == x1_)

    xx = tk.detokenize(x_)
    print(xx)
    assert(xx == x)

    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 == x1)

    # Adjust x1_ to have nul symbol.
    x1_[-1] = tk.nul_id
    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 == x1)

    x1_[-1] = 39  # an arbitrary token id.
    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 != x1)

    print("*** restart ***")
    tk = NeuralTokenizer(vocab, prefix_start=False, postfix_stop=False)

    x = [x[1] for x in train[:2]]
    print(x)
    x_ = tk.tokenize(x)
    print(x_)
    assert(len(x_[0]) == len(x[0]))
    assert(x_[0][0] != tk.start_id)
    assert(x_[0][-1] != tk.stop_id)
    assert(x_[0][1] == vocab.get(x[0][1]))

    # x[1] from row 1 of train.
    x1 = train[1][1]
    print(x1)
    x1_ = tk.tokenize(x1)
    print(x1_)
    assert(x[1] == x1)
    assert(x_[1] == x1_)

    xx = tk.detokenize(x_)
    print(xx)
    assert(xx == x)

    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 == x1)

    # Adjust x1_ to have nul symbol.
    x1_[-1] = tk.nul_id
    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 != x1)

    x1_[-1] = 39  # an arbitrary token id.
    xx1 = tk.detokenize(x1_)
    print(x1, x1_, xx1)
    assert(xx1 != x1)

def test_datasets():
    print("test datasets")

    train_data, test_data = get_data_for_language(
        language='English', form='Tokens')

    neuraldata = nd.NeuralData(train_data, test_data, val_split=0.2)
    assert(neuraldata.val_split == 0.2)
    assert(len(neuraldata.train_data) == len(train_data))
    assert(len(neuraldata.test_data) == len(test_data))
    vocab = Vocab([x[1] for x in train_data])
    assert(neuraldata.vocab.size == vocab.size)
    assert neuraldata.batch_size == settings.batch_size
    # assert neuraldata.token_maxlen == settings.token_maxlen

    train = neuraldata.train
    # assert train.length == len(train_data)
    # assert train.token_maxlen == neuraldata.token_maxlen
    assert len(train) == len(train_data)
    print("Type of dataset:", type(train))
    # print("Dir of dataset:", dir(train))
    assert isinstance(train, tf.data.Dataset)
    train_bat = train.batch(train.batch_size)
    print("train batch =", len(train_bat))
    assert len(train_bat) == math.ceil(len(train)/train.batch_size)
    train_lst = list(train_bat.unbatch().as_numpy_iterator())
    print("train =", len(train_lst))
    assert len(train_lst) == len(train)
    assert len(train_lst) == len(train_data)


    # assert(len(test_lst) == len(test_ds))

    # assert(len(train_lst) == neuraldata.train_len)
    # assert(len(test_lst) == neuraldata.test_len)
    fit = neuraldata.fit
    for x, y in fit.batch(fit.batch_size).take(1):
        print(x, y)
        print(x.shape, y.shape)
        assert x.shape[0] == fit.batch_size
        assert x.shape[1] == cfg.NeuralSettings().token_maxlen
        assert y.shape[0] == fit.batch_size
        assert y.shape[1] == cfg.NeuralSettings().token_maxlen


    # for i, (x, y) in enumerate(fit_lst):
    #     # print(len(x), len(y), i)
    #     if i == 0:
    #         print("step", i, "of ", steps)
    #         # print("initial batch", (x, y))
    #         # Check sequence length.
    #         print("x, y=", (x,y))
    #     if i == steps-1:
    #         print("Last batch", i)
    #         print("x, y=", (x,y))
    #         break
    #     print("step", i, "of ", steps)
    #     assert i == steps-1

    # val_lst = list(val.unbatch().as_numpy_iterator())
    # # print("fit =", len(fit_lst), "val =", len(val_lst))
    # assert(len(val_lst) == math.ceil(len(train_ds) * 0.2))
    # assert(len(fit_lst) == len(train_ds) - math.ceil(len(train_ds) * 0.2))
    # assert(len(val_lst) + len(fit_lst) == len(train_lst))

    # assert not all(a == b for a, b in zip(train_lst[0][0], val_lst[0][0]))
    # # Test whether initial case the same betreen train and val.
    # # Sort should make unlikely that same.

    # # Verify all token sequences conform to start, stop, padding.
    # for tk_ids in train_lst:
    #     assert tk_ids[0][0] == vocab.start_id  # token_ids_x
    #     assert tk_ids[1][0] != vocab.start_id  # token_ids_y
    #     i = 1
    #     while (i < neuraldata.token_maxlen and tk_ids[0][i]
    #            not in [vocab.nul_id, vocab.start_id, vocab.stop_id]):
    #         i += 1
    #     if i < neuraldata.token_maxlen:
    #         assert tk_ids[0][i] == vocab.nul_id
    #         assert tk_ids[1][i-1] == vocab.stop_id
    #         assert tk_ids[1][i] == vocab.nul_id
    #         i += 1
    #     while (i < neuraldata.token_maxlen
    #            and tk_ids[0][i] == vocab.nul_id):
    #         i += 1
    #     assert i == neuraldata.token_maxlen


    # # Add tests for missing statements from testing.
    # fit, val = neuraldata.get_fit_val_datasets(train, val_split=0.0)
    # fit_lst = list(fit.unbatch().as_numpy_iterator())
    # assert val == None
    # # print("fit =", len(fit_lst))
    # assert len(fit_lst) == len(train_ds)

    # neuraldata = nd.NeuralData(train_ds, test_ds, val_split=50)
    # assert neuraldata.val_split == 50
    # train, test = neuraldata.get_token_datasets()
    # fit, val = neuraldata.get_fit_val_datasets(train)
    # fit_lst = list(fit.unbatch().as_numpy_iterator())
    # val_lst = list(val.unbatch().as_numpy_iterator())
    # assert len(val_lst) == 50
    # # print("fit =", len(fit_lst))
    # assert len(fit_lst) == len(train_ds) - 50


def test_generators():
    print("test generators")

    train_ds, test_ds = get_data_for_language(language='English', form='Tokens')
    print("len(train) =", len(train_ds), "len(test)", len(test_ds))

    print("--preparation")
    neuraldata = nd.NeuralDataGen(train_ds, test_ds, val_split=0.2)
    assert neuraldata.val_split == 0.2
    assert len(neuraldata.train_data) == len(train_ds)
    assert len(neuraldata.test_data) == len(test_ds)
    vocab = Vocab([x[1] for x in train_ds])
    assert neuraldata.vocab.size == vocab.size
    assert len(neuraldata.val_data) == math.ceil(len(train_ds) * 0.2)
    assert len(neuraldata.fit_data) == math.floor(len(train_ds) * (1.0-0.2))

    print("--test generator")
    train = neuraldata.train
    assert train is not None
    assert len(train) == len(train_ds)
    assert train.batch_size == settings.batch_size
    # assert train.token_maxlen == settings.token_maxlen

    gen = train.generate()
    # We round up to next batch, repeating some elements from epoch.
    steps = math.ceil(len(train)/train.batch_size)
    for i, (x, y) in enumerate(gen):
        print(len(x), len(y), i)
        if i == 0:
            print("step", i, "of ", steps)
            # print("initial batch", (x, y))
            # Check sequence length.
            # print("x=", x)
        if i == steps-1:
            print("Last batch", i)
            # print("x=", (x, y)
            break
    print("step", i, "of ", steps)
    assert i == steps-1


    print("--test generator - with even batch length")

    train_ds = train_ds[:1184]
    neuraldata = nd.NeuralDataGen(train_ds, test_ds, val_split=0.2)

    train = neuraldata.train
    assert train is not None
    assert len(train) == len(train_ds)
    assert train.batch_size == settings.batch_size
    # assert train.token_maxlen == settings.token_maxlen

    gen = train.generate()
    # We round up to next batch, repeating some elements from epoch.
    steps = math.ceil(len(train)/train.batch_size)
    for i, (x, y) in enumerate(gen):
        print(len(x), len(y), i)
        if i == 0:
            print("step", i, "of ", steps)
            # print("initial batch", (x, y))
        if i == steps-1:
            print("Last batch", i)
            break
    print("step", i, "of ", steps)
    # print("final batch", (x, y))
    assert i == steps-1

    print("Test other generators")
    neuraldata = nd.NeuralDataGen(train_ds, test_ds, val_split=0.2)
    test = neuraldata.test
    fit = neuraldata.fit
    val = neuraldata.val
    assert len(test_ds) == len(test)
    assert math.ceil(len(train_ds) * 0.2) == len(val)
    assert math.floor(len(train_ds) * 0.8) == len(fit)


# def test_data():
#     print('test_data')
#     settings = NeuralSettings(val_split=0.15)  # To assure we have defined val_len
#     data = NeuralDataDirect(training1, testing1, settings=settings)
#     tokens = data.get_data_tokens(data.fit)
#     assert len(tokens) == len(data.fit)
#     tokens_ids = data.get_tokens_ids(tokens)
#     assert len(tokens_ids) == len(tokens)
#     assert data.translate(tokens) == tokens_ids
#     tokens_ids_ = data.get_data_tokens_ids(data.fit)
#     assert tokens_ids == tokens_ids_

#     trainer = data.trainer
#     validator = data.validator
#     tester = data.tester
#     assert trainer.data_len == len(data.fit)
#     assert validator.data_len == len(data.val)
#     assert tester.data_len == len(data.testing)
#     assert trainer == data.trainer
#     assert validator == data.validator
#     assert tester == data.tester

#     # Iterate at least 1ce through generator.
#     gen = trainer.generate()
#     x, y = next(gen)
#     print("x=", x)
#     print("y=", y)
#     steps = trainer.data_len // trainer.batch_size
#     for i, (x, y) in enumerate(gen):
#         if i > steps:
#             break


# def test_data1():
#     print('test_data1')
#     # Without testing data and with val_split == 0.
#     settings = NeuralSettings(val_split=0.0)
#     # Unresoloved error - neural data does not receive val_split
#     # Set via settings for now.
#     data = NeuralDataDirect(training1, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.0

#     tokens = data.get_data_tokens(data.fit)
#     assert len(tokens) == len(data.fit)
#     tokens_ids = data.get_tokens_ids(tokens)
#     assert len(tokens_ids) == len(tokens)
#     assert data.translate(tokens) == tokens_ids
#     tokens_ids_ = data.get_data_tokens_ids(data.fit)
#     assert tokens_ids == tokens_ids_

#     trainer = data.trainer
#     validator = data.validator
#     tester = data.tester
#     assert trainer.data_len == len(data.fit)
#     assert validator == None
#     assert tester == None
#     assert trainer == data.trainer
#     assert validator == data.validator
#     assert tester == data.tester

#     # With argument val_split == 0.1 to make sure it overrides the val_split.
#     settings = NeuralSettings(val_split=0.1)
#     data = NeuralData(training1, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.1

#     data = NeuralData(training1, val_split=0.0, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.0

# def test_data_hot():
#     print('test_data')
#     settings = NeuralSettings(val_split=0.15)  # To assure we have defined val_len
#     data = NeuralDataDirectHot(training1, testing1, settings=settings)
#     tokens = data.get_data_tokens(data.fit)
#     assert len(tokens) == len(data.fit)
#     tokens_ids = data.get_tokens_ids(tokens)
#     assert len(tokens_ids) == len(tokens)
#     assert data.translate(tokens) == tokens_ids
#     tokens_ids_ = data.get_data_tokens_ids(data.fit)
#     assert tokens_ids == tokens_ids_

#     trainer = data.trainer
#     validator = data.validator
#     tester = data.tester
#     assert trainer.data_len == len(data.fit)
#     assert validator.data_len == len(data.val)
#     assert tester.data_len == len(data.testing)
#     assert trainer == data.trainer
#     assert validator == data.validator
#     assert tester == data.tester

#     # Iterate at least 1ce through generator.
#     gen = trainer.generate()
#     x, y = next(gen)
#     print("x=", x)
#     print("y=", y)
#     steps = trainer.data_len // trainer.batch_size
#     for i, (x, y) in enumerate(gen):
#         if i > steps:
#             break


# def test_data1_hot():
#     print('test_data1')
#     # Without testing data and with val_split == 0.
#     settings = NeuralSettings(val_split=0.0)
#     # Unresoloved error - neural data does not receive val_split
#     # Set via settings for now.
#     data = NeuralDataDirectHot(training1, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.0

#     tokens = data.get_data_tokens(data.fit)
#     assert len(tokens) == len(data.fit)
#     tokens_ids = data.get_tokens_ids(tokens)
#     assert len(tokens_ids) == len(tokens)
#     assert data.translate(tokens) == tokens_ids
#     tokens_ids_ = data.get_data_tokens_ids(data.fit)
#     assert tokens_ids == tokens_ids_

#     trainer = data.trainer
#     validator = data.validator
#     tester = data.tester
#     assert trainer.data_len == len(data.fit)
#     assert validator == None
#     assert tester == None
#     assert trainer == data.trainer
#     assert validator == data.validator
#     assert tester == data.tester
#     assert trainer.label_size == 2

#     # With argument val_split == 0.1 to make sure it overrides the val_split.
#     settings = NeuralSettings(val_split=0.1)
#     data = NeuralDataDirectHot(training1, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.1

#     data = NeuralDataDirectHot(training1, val_split=0.0, settings=settings)
#     print(f"val_split {data.val_split}")
#     assert data.val_split == 0.0


# def test_data_for_back():
#     print('test_data_for_back')
#     settings = NeuralSettings(val_split=0.15)  # To assure we have defined val_len
#     data = NeuralDataForBack(training1, testing1, settings=settings)
#     tokens = data.get_data_tokens(data.fit)
#     assert len(tokens) == len(data.fit)
#     tokens_ids = data.get_tokens_ids(tokens)
#     assert len(tokens_ids) == len(tokens)
#     assert data.translate(tokens) == tokens_ids
#     tokens_ids_ = data.get_data_tokens_ids(data.fit)
#     assert tokens_ids == tokens_ids_

#     trainer = data.trainer
#     validator = data.validator
#     tester = data.tester
#     assert trainer.data_len == len(data.fit)
#     assert validator.data_len == len(data.val)
#     assert tester.data_len == len(data.testing)
#     assert trainer == data.trainer
#     assert validator == data.validator
#     assert tester == data.tester

#     # Iterate at least 1ce through generator.
#     gen = trainer.generate()
#     [x, x_back], [y, y_back] = next(gen)
#     print("x=", x)
#     print("x_back=", x_back)
#     print("y=", y)
#     print("y_back=", y_back)
#     steps = trainer.data_len // trainer.batch_size
#     for i, ([x, x_back], [y, y_back]) in enumerate(gen):
#         if i > steps:
#             break


# if __name__ == "__main__":
#     # test_vocab()
#     # test_data()
#     # test_data1()
#     # test_data_hot()
#     # test_data1_hot()
#     test_data_for_back()
