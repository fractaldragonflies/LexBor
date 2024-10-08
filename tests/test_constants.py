#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:58:41 2024

@author: johnmiller


Test transformer_decoder.py

All test functions begin with 'test_'

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_constants.py -—cov=lexbor.constants

Or to save as html report:
$ pytest tests/test_constants.py -—cov=lexbor.constants --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_constants.py -—cov=lexbor.constants --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_constants.py::test_input_format -—cov=lexbor.constants--cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests,
or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

Add -s to capture output.

"""

from lexbor.constants import DataFmtConst, detect_input_fmt

import pytest

def test_input_format():
    print("test - test input format")
    """
        Format is either [[IPA_segments]], or [[str, [IPA_segments], int]].
    """
    test_SEG = [['a', 'b', 'c'],
                ['ph', 'ts', 'e', 'k']]
    test_STD = [['A1', ['a', 'b', 'c'], 0],
                ['B2', ['ph', 'ts', 'e', 'k'], 1]]
    test_UNK = [['A1', ['a', 'b', 'c']],
                ['B2', ['ph', 'ts', 'e', 'k']]]

    # def is_list_str(ele):
    #     return (ele and isinstance(ele, list)
    #             and all(isinstance(s, str) for s in ele))

    # def detect_input_fmt(data:list):
    #     row = data[0]  # get initial row to detect file format.
    #     if len(row) == 3 and is_list_str(row[1]):
    #         fmt = DataFmtConst.STD
    #     elif is_list_str(row):
    #         fmt = DataFmtConst.SEG
    #     else:
    #         fmt = DataFmtConst.UNK
    #     return fmt

    print("SEG", detect_input_fmt(test_SEG))
    print("STD", detect_input_fmt(test_STD))
    print("UNK", detect_input_fmt(test_UNK))
    assert detect_input_fmt(test_SEG) == DataFmtConst.SEG
    assert detect_input_fmt(test_STD) == DataFmtConst.STD
    assert detect_input_fmt(test_UNK) == DataFmtConst.UNK
