#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:45:29 2024

@author: johnmiller


Generate histogram plots showing disribution of cross-entropies.
"""

# Import Python standard libraries
import statistics

# Import 3rd-party libraries
from matplotlib import pyplot as plt
import numpy as np
import math


def graph_distribution(values1=None, values2=None, output_path=None, **kwargs):

    title = kwargs.get("title", None)
    label1 = kwargs.get("label1", "")
    label2 = kwargs.get("label2", "")
    xlabel = kwargs.get("xlabel", "Cross-entropy")
    ylabel = kwargs.get("ylabel", "Frequency")
    legloc = kwargs.get("loc", "upper right")
    graph_limit = kwargs.get("graph_limit", None)


    # Get upper and lower graph limits based on third most extreme values.
    values = sorted(values1 + (values2 if values2 else []))
    upper_limit = (graph_limit if graph_limit is not None
                   else math.ceil(values[-3]))
    lower_limit = min(0, math.floor(values[3]))
    # print("Limits", upper_limit, lower_limit)

    # Set frame horizontal for this measure.
    bins = np.linspace(lower_limit, upper_limit, 60)
    plt.figure(figsize=(8, 5))

    cnt1 = f"{len(values1):6d}"
    avg1 = f"{statistics.mean(values1):6.3f}"
    std1 = f"{statistics.stdev(values1):6.3f}"
    plt.hist(
        values1,
        bins,
        alpha=0.65,
        label=f"{label1}$(n={cnt1}, \\mu={avg1}, \\sigma={std1})$",
        color="blue",
    )
    if values2:
        cnt2 = f"{len(values2):6d}"
        avg2 = f"{statistics.mean(values2):6.3f}"
        std2 = f"{statistics.stdev(values2):6.3f}"
        plt.hist(
            values2,
            bins,
            alpha=0.65,
            label=f"{label2}$(n={cnt2}, \\mu={avg2}, \\sigma={std2})$",
            color="red",
        )

    plt.grid(axis="y", alpha=0.8)
    plt.legend(loc=legloc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if output_path:
        plt.savefig(output_path, dpi=600)
    plt.show()
    plt.close()