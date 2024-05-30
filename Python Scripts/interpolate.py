#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: Interpolation class used to Interpolate missing values
        using time based linear interpolation.

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""
import logging
import numpy as np

class Interpolate:
    """
    A custom class for performing interpolation on time series data.

    Parameters:
        input_data (pandas.Series): The input time series data containing missing values.
        propagate_log (bool, optional): Flag to propagate logging messages. Defaults to True.
    """


    def __init__(self, input_data, propagate_log = True):
        """
        Class constructor. Initializes an instance of Interpolate.

        Args:
            input_data (pandas.Series): The input time series data containing missing values.
            propagate_log (bool, optional): Flag to propagate logging messages. Defaults to True.
        """

        self.input_data = input_data
        self.imputed_data = None
        self.interpolation_logger = logging.getLogger(type(self).__name__)
        self.interpolation_logger.propagate = propagate_log

    def impute(self):
        """
        Perform interpolation to fill missing values in the input data.
        """
        self.interpolation_logger.info("Starting Interpolation Method.")
        for time in self.input_data.index:
            if np.isnan(self.input_data.loc[time]):
                self.fill_gap(time)
        self.imputed_data = self.input_data
        self.interpolation_logger.info("Finished Interpolation Method.")

    def fill_gap(self, time):
        """
        Fill the missing values in a gap identified by the given time.

        Args:
            time: The time index indicating the start of the gap.
        """
        first_index = self.input_data.index.get_loc(time)
        last_index = self.find_last_index(first_index)

        for index in (range(((last_index - first_index) + 1) // 2)) if first_index != last_index else range(1):
            lr_index = first_index + index
            start_time_lr, start_value_lr = self.get_start_time_value(lr_index, -1)
            end_time_lr, end_value_lr = self.get_end_time_value(last_index - index, +1)
            value_time_lr = self.input_data.index[lr_index]
            interpolated_value_lr = self.interpolate_value(value_time_lr, start_time_lr, start_value_lr, end_time_lr, end_value_lr)
            self.input_data.iloc[lr_index] = round(interpolated_value_lr, 2)

            rl_index = last_index - index
            start_time_rl, start_value_rl = self.get_start_time_value(rl_index, +1)
            end_time_rl, end_value_rl = self.get_end_time_value(first_index + index,  -1)
            value_time_rl = self.input_data.index[rl_index]
            interpolated_value_rl = self.interpolate_value(value_time_rl, start_time_rl, start_value_rl, end_time_rl, end_value_rl)
            self.input_data.iloc[rl_index] = round(interpolated_value_rl, 2)


    def find_last_index(self, first_index):
        """
        Find the index of the last non-NaN value following a gap starting at the given index.

        Args:
            first_index: The index indicating the start of the gap.

        Returns:
            int: The index of the last non-NaN value in the gap.
        """
        index_inc = 0
        while np.isnan(self.input_data.iloc[first_index + index_inc]):
            index_inc += 1
        return first_index + index_inc -1

    def get_start_time_value(self, index, direction_indicator):
        """
        Get the time and value of the neighboring data point in the specified direction.

        Args:
            index: The index of the current data point.
            direction_indicator: An indicator (-1 or +1) specifying the direction.

        Returns:
            tuple: A tuple containing the time and value of the neighboring data point.
        """
        start_time = self.input_data.index[index + direction_indicator]
        start_value = self.input_data.iloc[index + direction_indicator]
        return start_time, start_value

    def get_end_time_value(self, index, direction_indicator):
        """
        Get the time and value of the neighboring data point in the specified direction.

        Args:
            index: The index of the current data point.
            direction_indicator: An indicator (-1 or +1) specifying the direction.

        Returns:
            tuple: A tuple containing the time and value of the neighboring data point.
        """
        end_time = self.input_data.index[index + direction_indicator]
        end_value = self.input_data.iloc[index + direction_indicator]
        return end_time, end_value

    def interpolate_value(self, value_time, start_time, start_value, end_time, end_value):
        """
        Perform linear interpolation to estimate the value at the specified time.

        Args:
            value_time: The time at which the value is to be interpolated.
            start_time: The time of the start data point for interpolation.
            start_value: The value of the start data point for interpolation.
            end_time: The time of the end data point for interpolation.
            end_value: The value of the end data point for interpolation.

        Returns:
            float: The interpolated value.
        """
        return start_value + (end_value - start_value) * ((value_time - start_time) / (end_time - start_time))

    def get_imputed_values(self):
        """
        Get the imputed values after performing interpolation.

        Returns:
            pandas.Series: The imputed time series data.
        """
        return self.imputed_data

