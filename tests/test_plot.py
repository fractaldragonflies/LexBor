#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:08:56 2024

@author: johnmiller


Test plot.py

All test functions begin with 'test_'

Install pytest and pytest-cov with pip

Save test files to ./tests

Run test file:
$ pytest tests/test_plot.py -—cov=lexbor.plot

Or to save as html report:
$ pytest tests/test_plot.py -—cov=lexbor.plot --cov-report=html

Or to report line numbers of missing coverage:
$ pytest tests/test_plot.py -—cov=lexbor.plot --cov-report term-missing

Or to perform a specific test:
$ pytest tests/test_plot.py::test_graph_distribution -—cov=lexbor.plot --cov-report term-missing

File names can come last instead of first in the argument list.

Add the -rP argument to print output for passing tests,
or -rPx for passing and failed tests.
Otherwise default is -rx for failed tests only.

Add -s to capture output.

"""

from lexbor.plot import graph_distribution
from pathlib import Path
import lexbor.config as cfg

output_path = Path(cfg.BaseSettings().output_path).resolve()

def test_graph_distribution():
    print("test - graph distribution")

    values = [0.57, 1.34, 3.12, 1.56, 0.69, 2.55, 2.83, 3.98,
              3.1, 2.9, 2.54, 1.76, 1.86, 0.97]
    # Histogram of one series without title or sequence labels
    # with legend at upper right. Graph range [0, 4].
    graph_distribution(values)
    # Same graph with legend at upper left.
    graph_distribution(values, loc="upper left")
    # Same graph with legend at upper right and sequence "All fit".
    # Title "All fit". Should save copy to "dist_plot.png".
    path_out = output_path / "dist_plot.png"
    graph_distribution(values, output_path=path_out,
                       title="Test distribution", label1="All fit")

    values1 = [0.57, 1.34, 3.12, 1.56, 0.69, 2.55, 2.83, 3.98,
              3.1, 2.9, 2.54, 1.76, 1.86, 0.97, 5.3, -0.31]
    # Same graph with legend at upper right and sequence "All fit".
    # Title "All fit". Should save copy to "dist_plot.png".
    # Even though values < 0 and >4 they are censored on the graph.
    graph_distribution(values1, output_path=path_out,
                       title="Test distribution", label1="All fit")

    values2 = [2.57, 3.34, 5.12, 3.56, 2.69, 4.55, 4.83, 5.98,
              4.1, 4.9, 4.54, 3.76, 3.86, 0.97, 7.3, -0.31]
    graph_distribution(values1, values2, output_path=path_out,
                       title="Test 2 distributions",
                       label1="One fit", label2="Two fit",
                       ylabel="Proportion")
