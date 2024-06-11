#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 12 16:03 2023

@author: Ethan Masters

Purpose: ARIMA Custom Class

Python Version: Python 3.10.11 (main, Apr 20 2023, 13:59:00)
"""

import logging
import numpy as np
import pandas as pd
import os
import json
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


class ARIMA:
    def __init__(self, var_name = None, p = None, d = None, q = None, propagate_log = True):
        """
        Initializes ARIMA class with provided parameters.

        Args:
            var_name (str): Name of the variable.
            p (int): Order of the autoregressive model.
            d (int): Degree of differencing.
            q (int): Order of the moving average model.
            propagate_log (bool): Flag to propagate logging messages.

        """
        self.p = p
        self.d = d
        self.q = q
        self.var_name = var_name
        self.coefficients = None
        self.arima_logger = logging.getLogger(type(self).__name__)
        self.arima_logger.propagate = propagate_log

    def fit(self, data):
        """
        Fit ARIMA model to the given data.

        Args:
            data (numpy array): Time series data.

        """
        self.d = self.adfuller_testing(data)
        self.pre_difference = data[-self.d:]
        self.differenced_data = self.difference(data)
        self.intercept = self.estimate_intercept()
        self.ar_coefficients = (self.estimate_ar_coefficients(self.differenced_data) if self.p else 0)
        self.ma_coefficients = (self.estimate_ma_coefficients(self.differenced_data) if self.q else 0)
    
    def estimate_intercept(self):
        """
        Estimate the intercept term of the model.

        Returns:
            float: Estimated intercept value.

        """
        return np.mean(self.differenced_data)

    def difference(self, data):
        """
        Difference the data to make it stationary.

        Args:
            data (numpy array): Time series data.

        Returns:
            numpy array: Differenced data.

        """
        differenced_data = np.diff(data, n = self.d)
        return differenced_data
    
    def estimate_ar_coefficients(self, data):
        """
        Estimate AR coefficients using ordinary least squares (OLS).

        Args:
            data (numpy array): Time series.

        Returns:
            numpy array: Estimated AR coefficients.

        """
        N = len(data)
        Y = data[self.p:]
        A = np.zeros((N - self.p, self.p))

        for row_idx in range(self.p, N):
            A[row_idx - self.p] = data[row_idx - self.p:row_idx]

        At = np.transpose(A)
        AtY = np.dot(At, Y)
        AtA = np.dot(At, A)
        AtA_inv = np.linalg.inv(AtA)
        estimated_parameter_vector = np.dot(AtA_inv, AtY)
        return estimated_parameter_vector
    
    def estimate_ma_coefficients(self, data):
        """
        Estimate MA coefficients using ordinary least squares (OLS).

        Args:
            data (numpy array): Time series data.

        Returns:
            numpy array: Estimated MA coefficients.

        """
        N = len(data)
        errors = np.zeros(N)

        for i in range(self.p, N):
            ar_term = np.dot(data[i-self.p:i], self.ar_coefficients) if self.p > 0 else 0
            errors[i] = data[i] - ar_term

        Y = errors[self.q:]
        A = np.zeros((N - self.q, self.q))

        for row_idx in range(self.q, N):
            A[row_idx - self.q] = errors[row_idx - self.q:row_idx]

        At = np.transpose(A)
        AtY = np.dot(At, Y)
        AtA = np.dot(At, A)
        AtA_inv = np.linalg.inv(AtA)
        estimated_parameter_vector = np.dot(AtA_inv, AtY)
        return estimated_parameter_vector
    
    def forecast(self, steps):
        """
        Forecast future values using the fitted ARIMA model.

        Args:
            data (numpy array): Time series.
            steps (int): Number of steps to forecast into the future.

        Returns:
            numpy array: Forecasted values.

        """
        forecast_values = np.zeros(steps)
        for i in range(steps):
            forecast_value = self.forecast_step()
            forecast_value = self.invert_difference(forecast_value, i)
            forecast_values[i] = forecast_value
            self.differenced_data = np.concatenate([self.differenced_data[1:], [forecast_value]])
        return forecast_values

    def invert_difference(self, forecast_value, step):
        """
        Invert differencing to obtain actual forecasted values.

        Args:
            forecast_value (float): Forecasted value.
            step (int): Step of the forecast.

        Returns:
            float: Inverted forecasted value.

        """
        if self.d == 0: return forecast_value
        value = forecast_value + sum(self.pre_difference[step:])
        self.differenced_data = np.append(self.differenced_data, forecast_value)
        return value

    def forecast_step(self):
        """
        Forecast the next single step using the fitted ARIMA model.

        Returns:
            float: Forecasted value for the next step.

        """
        if len(self.differenced_data) < self.p or len(self.differenced_data) < self.q:
            raise ValueError("Insufficient data for forecasting")
        ar_term = np.dot(self.differenced_data[-self.p:], self.ar_coefficients[:self.p])
        ma_term = np.dot(self.differenced_data[-self.q:], self.ma_coefficients[:self.q])
        forecast_value = ar_term + ma_term
        return round(forecast_value, 4)
    
    @staticmethod
    def adfuller_testing(data):
        """
        Perform augmented Dickey-Fuller test to determine the order of differencing.

        Args:
            data (numpy array): Time series data.

        Returns:
            int: Order of differencing.

        """
        for d in range(4):
            result = adfuller(np.diff(data, n=d))
            if result[0] < 0.05:
                return d
        return 0
    
    @staticmethod
    def plot_autocorr(data):
        """
        Plot autocorrelation function (ACF) for the given data.

        Args:
            data (numpy array): Time series data.

        """
        plot_acf(data)
        plt.title('Autocorrelation Function (ACF)')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()
    
    @staticmethod
    def plot_partial_autocorr(data):
        """
        Plot partial autocorrelation function (PACF) for the given data.

        Args:
            data (numpy array): Time series data.

        """
        plot_pacf(data)
        plt.title('Partial Autocorrelation Function (PACF)')
        plt.xlabel('Lag')
        plt.ylabel('Partial Autocorrelation')
        plt.show()

    def get_state(self):
        """
        Get the state of the ARIMA model.

        Returns:
            dict: Dictionary containing the model state.

        """
        state_dict = {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'var_name': self.var_name,
            'coefficients': self.coefficients
        }
        return state_dict


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'arima_log.log')
    logging.basicConfig(filename = file_path,
                        level = logging.INFO,
                        format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    var_name = 'DirectIR'
    logger.info(f"Starting Arima {var_name} Model...")

    file_name = 'training_data_h.csv'
    file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file_name)
    with open(file_path, 'r') as f:
        input_data_dict = json.load(f)
    

    num_missing = 2
    data = np.array(input_data_dict[var_name])
    diff_list = []
    for arr in data[20:25]:
        arr = np.array(arr)
        if np.all(arr == 0): continue
        X = arr[:-num_missing]
        y = pd.Series(arr[-num_missing:])
        
        diff_list += [ARIMA.adfuller_testing(arr)]
        ARIMA.plot_autocorr(arr)
        ARIMA.plot_partial_autocorr(arr)

        arima_model = ARIMA(var_name = var_name,
                        p = 12,
                        d = 1,
                        q = 12)
        arima_model.fit(X)
        forecast_values = pd.Series(arima_model.forecast(X, steps=num_missing))
    
    unique_values = set(diff_list)
    for value in unique_values:
        count = diff_list.count(value)
        print("Number of", value, ":", count)

    logger.info(f"ARIMA {var_name} Model configuration:\n{arima_model.get_state()}")
    logger.info("------------- Error Statistics ------------")
    logger.info("Complete.")
    logger.info("\n# ====================================================================== #\n"
                + "# =============================== New Run ============================== #\n"
                + "# ====================================================================== #\n")