#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: General testing script

Python Version: Python 3.10.11 
"""

import pandas as pd
import numpy as np

def find_consecutive_nan_indices_with_timestamp(time_series):
    nan_indices = np.where(pd.isna(time_series))[0]
    if len(nan_indices) == 0:
        return []
    
    diff = np.diff(nan_indices)
    split_indices = np.where(diff != 1)[0]
    
    consecutive_segments = []
    start_index = nan_indices[0]
    for split_index in split_indices:
        end_index = nan_indices[split_index]
        segment = list(range(start_index, end_index + 1))
        consecutive_segments += [segment]
        start_index = nan_indices[split_index + 1]
    
    last_segment = list(range(start_index, nan_indices[-1] + 1))
    consecutive_segments += [last_segment]

    timestamps_consecutive_segments = [[time_series.index[index] for index in segment] for segment in consecutive_segments]
    print(timestamps_consecutive_segments)


# Create a time series with NaN gaps and timestamp index
index = pd.date_range(start='2024-05-01', periods=17, freq='20S')
time_series_with_index = pd.Series([np.nan, 1, 2, 3, np.nan, np.nan, 6, np.nan, 8, 9, 10, np.nan, np.nan, np.nan, 14, np.nan, 16], index=index)

# Find consecutive NaN gaps and return timestamp labels
consecutive_nan_timestamps = find_consecutive_nan_indices_with_timestamp(time_series_with_index)
