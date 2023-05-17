# BSD 3-Clause License
#
# This file is part of the PNEC project.
# https://github.com/tum-vision/pnec
#
# Copyright (c) 2022, Dominik Muhle.
# All rights reserved.

from typing import Optional, List
import numpy as np
import pandas as pd


def HighlightExtrema(data, extrema, highlights, precision):
    for extremum, highlight in zip(extrema, highlights):
        if data == extremum:
            return highlight % data
    return precision % data


def TableHighlighting(
    dataframe,
    minimum=True,
    axis=0,
    highlighting=["\\bfseries %.3f", "\\underline{%.3f}"],
    precision="%.3f",
    ignore_idx: Optional[List[int]] = None,
):
    if not axis in [0, 1]:
        print("axis has to be either 0 (row wise) or 1 (column wise)")
        return None

    num_highlights = len(highlighting)
    if axis == 0:
        # non_ignored_inx = [i for i in range(dataframe.shape[0]) if i not in ignore_idx]
        for k in range(dataframe.shape[0]):
            row = dataframe.iloc[k, :].to_numpy()
            if ignore_idx is not None:
                filtered_row = np.delete(row, ignore_idx, 0)
            else:
                filtered_row = row
            row_sorted = np.sort(filtered_row, axis=0)
            if minimum:
                extrema = row_sorted[:num_highlights]
            else:
                extrema = np.flip(row_sorted, axis=0)[:num_highlights]
            dataframe.iloc[k, :] = dataframe.iloc[k, :].apply(
                lambda data: HighlightExtrema(data, extrema, highlighting, precision)
            )
    if axis == 1:
        if ignore_idx is None:
            ignore_idx = []
        non_ignored_inx = [i for i in range(dataframe.shape[1]) if i not in ignore_idx]
        for k in dataframe.columns:
            if minimum:
                extrema = dataframe[k].nsmallest(num_highlights)
            else:
                extrema = dataframe[k].nlargest(num_highlights)
            dataframe[k] = dataframe[k].apply(lambda data: HighlightExtrema(data, extrema, highlighting, precision))

    return dataframe
