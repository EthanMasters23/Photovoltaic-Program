#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on Apr 12 16:03 2024

@author: Ethan Masters

Description: Contains modules to be used in the evaluation of 
    different methods for the purposes of imputing
    missing values in the PV Solar Data Project.

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import random
import logging

# == helper functions == #
from pv_modules import PVModules

# = set for reproducibility = #
# random.seed(42)

class ImputationModules:
    """
    A class for handling imputation operations on PV modules data.

    Parameters:
        FILE_TYPE (str, optional): The type of file to process. Default is None.
        YEAR (str, optional): The year of the data to process. Default is None.
        MONTH (str, optional): The month of the data to process. Default is None.
    """
    def __init__(self, FILE_TYPE = None, propagate_log = True):
        """
        Class constructor. Initializes an instance of ImputationModules.

        Args:
            FILE_TYPE (str, optional): The type of file to process. Default is None.
            YEAR (str, optional): The year of the data to process. Default is None.
            MONTH (str, optional): The month of the data to process. Default is None.
        """
        self.FILE_TYPE = FILE_TYPE
        self.imputation_data_frame = pd.DataFrame()
        self.test_imputation_data_frame = pd.DataFrame()
        self.meteoblue_data_frame = pd.DataFrame()
        self.calculations_data_frame = pd.DataFrame()
        self.imputation_data_dict = dict()
        self.error_data_frame = pd.DataFrame()

        self.month_of_data = int()
        self.initial_month = int()
        self.initial_year = int()
        self.initial_cond = bool()
        self.test_gaps = dict()
        self.path = str()
        self.imputation_logger = logging.getLogger(type(self).__name__)
        self.imputation_logger.propagate = propagate_log

    def clean_data_frame(self):
        '''
        Cleans and processes a dataframe based on the given file type.
        
        Args:
            path (list): List of file paths where the dataframe is stored.
            file (str): Type of file ('Irradiance', 'Deger', 'Fixed').
            
        Returns:
            pandas.DataFrame: Cleaned and processed dataframe.
        '''
        pv_module = PVModules(data_frame = pd.read_csv(self.path, sep = "\t|,", engine = 'python'),
                              FILE_TYPE = self.FILE_TYPE,
                              propagate_logger = False)
        # === In case a file isn't stored properly or empty === #
        if pv_module.get_data_frame().empty:
            raise Exception("Loaded an empty dataframe")
        # ==== reshaping df for timestap & adjusted headers ==== #
        pv_module.reshape_df()

        if self.FILE_TYPE == 'Irradiance':
            # === Removing Values for Irradiance === #
            pv_module.clean_irradiance_values()
        else:
            # === Removing Values for Deger & Fixed === #
            pv_module.clean_deger_fixed_values()

        # ==== Using PvLib to set nightime values === #
        pv_module.set_night_values()
        # ===  resample df for 20s frequency === #
        pv_module.resample_df()
        impute_data_frame = pv_module.get_data_frame().copy()
        impute_data_frame = impute_data_frame.drop(['GlobalIR'], axis=1)
        self.imputation_data_frame = impute_data_frame

    def simulate_missing_values(self, gap_length):
        '''
        Simulates missing values in the dataframe for a specified gap length.
        
        Args:
            df (pandas.DataFrame): The input dataframe.
            gap_length (int): Length of the gap in seconds for which missing values are to be simulated.
            
        Returns:
            pandas.DataFrame: The dataframe with simulated missing values.
            pandas.DataFrame: A copy of the original dataframe.
        '''
        self.imputation_data_frame = self.imputation_data_frame.dropna()
        self.test_imputation_data_frame = self.get_imputation_data_frame()
        if not gap_length:
            return
        
        index_list = PVModules.remove_nighttime_values(self.get_test_data_frame()).index

        if gap_length > 20000:
            while True:
                start_index = random.choice(index_list)
                end_index = start_index + pd.Timedelta(gap_length, 'S')
                if end_index < index_list[-1]:
                    break
            self.test_imputation_data_frame.loc[start_index:end_index,:] = np.nan
        else:
            while True:
                start_index = random.choice(index_list)
                end_index = start_index + pd.Timedelta(gap_length, 'S')
                if start_index.day == end_index.day and end_index < index_list[-1]:
                    break
            self.test_imputation_data_frame.loc[start_index:end_index,:] = np.nan

        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets')
        self.test_imputation_data_frame.to_csv(os.path.join(file_path, 'test_data_(temp).csv'))
        self.imputation_data_frame.to_csv(os.path.join(file_path, 'validation_data_(temp).csv'))

    def map_nan_gap_indices(self):
        '''
        Maps the indexes of NaN gaps in the dataframe for a specific test month.
        
        Args:
            df (pandas.DataFrame): The input dataframe.
            test_month (int): The test month for which NaN gaps are to be mapped.
            
        Returns:
            dict: A dictionary where keys are column names and values are lists of indexes 
                representing NaN gaps in the corresponding columns for the specified month.
        '''
        for col in self.test_imputation_data_frame.columns:
            nan_indices = []
            for date in self.test_imputation_data_frame.index:
                if date.month != int(self.month_of_data) or not np.isnan(self.test_imputation_data_frame.loc[date, col]):
                    continue
                nan_indices += [date]
            if nan_indices:
                self.test_gaps[col] = nan_indices     
        first_key = list(self.test_gaps.keys())[0] 
        self.imputation_logger.info(f"Imputing for {len(self.test_gaps[first_key])} gap(s) start : {self.test_gaps[first_key][0]} end: {self.test_gaps[first_key][-1]}")

    def missing_data_edge_case(self):
        '''
        Checks for missing data at the beginning and end of the month in the dataframe,
        and loads data from the surrounding months if necessary to fill the gaps.
        
        Args:
            df (pandas.DataFrame): The input dataframe.
            base_df (pandas.DataFrame): The base dataframe to which missing data is joined.
            file (str): Type of file ('Irradiance', 'Deger', 'Fixed').
            
        Returns:
            pandas.DataFrame: The dataframe with missing data filled and sorted by index.
            pandas.DataFrame: The base dataframe with missing data joined and sorted by index.
            int: The month index of the original dataframe.
        '''
        
        self.initial_month = self.test_imputation_data_frame.index[0].month
        self.month_of_data = self.test_imputation_data_frame.index[0].month
        self.initial_year = self.test_imputation_data_frame.index[0].year

        beg_ind = False
        end_ind = False

        beg_threshold = int(len(self.test_imputation_data_frame) / self.test_imputation_data_frame.iloc[-1].name.day * 3)
        end_threshold = int(len(self.test_imputation_data_frame) / self.test_imputation_data_frame.iloc[-1].name.day * (self.test_imputation_data_frame.iloc[-1].name.day - 3))

        percentage_thershold = 5

        copy_imputation_data_frame = self.get_imputation_data_frame()

        # Check for missing data in the beginning of the month
        if self.test_imputation_data_frame.iloc[0:beg_threshold, :].isna().sum().sum() / self.test_imputation_data_frame.iloc[0:beg_threshold, :].size * 100 > percentage_thershold:
            if self.initial_month == 9 and self.initial_year == 2021:
                raise Exception("September 2021 is the first month observed, can't join prior month.")
            self.find_surrounding_month()
            self.path = PVModules(
                self.FILE_TYPE,
                str(self.initial_year),
                self.initial_month
            ).path_list[0]
            self.clean_data_frame()
            beg_data_frame = self.get_imputation_data_frame()
            beg_ind = True

        # Check for missing data at the end of the month
        if self.test_imputation_data_frame.iloc[end_threshold:-1, :].isna().sum().sum() / self.test_imputation_data_frame.iloc[end_threshold:-1, :].size * 100 > percentage_thershold:
            if self.initial_month == 5 and self.initial_year == 2023:
                raise Exception("February 2023 is the last month observed, can't join following month.")
            self.find_surrounding_month()
            self.path = PVModules(
                self.FILE_TYPE,
                str(self.initial_year),
                self.initial_month
            ).path_list[0]

            self.clean_data_frame()
            end_data_frame = self.get_imputation_data_frame()
            end_ind = True

        self.imputation_data_frame = copy_imputation_data_frame

        if beg_ind and end_ind:
            self.test_imputation_data_frame = pd.concat([beg_data_frame, self.test_imputation_data_frame, end_data_frame], axis = 0, ignore_index = False)
            self.imputation_data_frame =  pd.concat([beg_data_frame, self.imputation_data_frame, end_data_frame], axis = 0, ignore_index = False)
        elif beg_ind:
            self.test_imputation_data_frame = pd.concat([beg_data_frame, self.test_imputation_data_frame], axis = 0, ignore_index = False)
            self.imputation_data_frame =  pd.concat([beg_data_frame, self.imputation_data_frame], axis = 0, ignore_index = False)
        elif end_ind:
            self.test_imputation_data_frame = pd.concat([self.test_imputation_data_frame, end_data_frame], axis = 0, ignore_index = False)
            self.imputation_data_frame =  pd.concat([self.imputation_data_frame, end_data_frame], axis = 0, ignore_index = False)

    def load_meteoblue_data(self):
        """
        Fill missing values in the test_df DataFrame using hourly data from the ml_df DataFrame.

        Args:
            test_df (pandas.DataFrame): The DataFrame containing the test data.
            file (str): The type of file for which missing values need to be filled. 
                        Valid options are "Irradiance" or any other file type.

        Returns:
            tuple: A tuple containing the modified test_df DataFrame and the ml_df DataFrame.
        """
        meteo_file = 'meteo_data_total_clean_4D_4Y'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'{meteo_file}.csv')
        self.meteoblue_data_frame = pd.read_csv(file_path, index_col = 0)
        self.meteoblue_data_frame.index = pd.to_datetime(self.meteoblue_data_frame.index)
        self.meteoblue_data_frame = self.meteoblue_data_frame.loc[self.test_imputation_data_frame.index[0]:self.test_imputation_data_frame.index[-1],:]
        self.meteoblue_data_frame = self.meteoblue_data_frame[self.meteoblue_data_frame.index.isin(self.test_imputation_data_frame.index)]

        # if file == "Irradiance":
        #     pass
        #     test_df['DirectIR'] = test_df['DirectIR'].fillna(hour_ml_df['Direct Shortwave Radiation (W/m²) (sfc)'])
        #     test_df['DiffuseIR'] = test_df['DiffuseIR'].fillna(hour_ml_df['Diffuse Shortwave Radiation (W/m²) (sfc)'])
        #     test_df['Temperature'] = test_df['Temperature'].fillna(hour_ml_df['Temperature (°C) (2 m elevation corrected)'])
        #     test_df['WindSpeed'] = test_df['WindSpeed'].fillna(hour_ml_df['Wind Speed (km/h) (10 m)'])
        # else:
        #     print(f"No available data to fill hourly gaps for file type: {file}")

    def find_surrounding_month(self):
        """
        Get the surrounding month and year based on the input month, year, and condition.

        Args:
            month (int): The input month (1-12).
            year (int): The input year.
            cond (str): The condition indicating the desired surrounding month. Possible values are 'prev' or 'next'.

        Returns:
            tuple: A tuple containing the surrounding month and year.
        """
        month_dict = {0:'dec', 13:'jan', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec', 1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul'}
        if self.initial_cond == 'prev':
            if self.initial_month == 1:
                self.initial_year -= 1
            self.initial_month = month_dict[self.initial_month-1]
        else:
            if self.initial_month == 12:
                self.initial_year += 1
            self.initial_month = month_dict[self.initial_month+1]

    @staticmethod
    def find_month(month):
        """
        Get the name of the month based on its numerical representation.

        Args:
            month (int): The numerical representation of the month.

        Returns:
            str: The name of the month.
        """
        month_dict = {0:'dec', 13:'jan', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec', 1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul'}
        return month_dict[month]

    # ====================================================================== #
    # ========================= Imputation Methods ========================= #
    # ====================================================================== #

    def interpolation(self):
        """
        Perform interpolation for missing values in the dataframe.
        """
        self.calculations_data_frame = self.get_test_data_frame()
        for col in self.calculations_data_frame.columns:
            if not self.calculations_data_frame[col].isna().sum(): continue
            self.calculations_data_frame[col] = self.calculations_data_frame[col].interpolate(method='time', limit_direction='both')
                
        self.calculate_imputation_errors()
        self.imputation_data_dict['Interpolation'] = self.error_data_frame.to_dict()

    def interpolation_method(self):
        """
        Perform interpolation using custom class for missing values in the dataframe.
        """
        from interpolate import Interpolate
        self.calculations_data_frame = self.get_test_data_frame()
        for col in self.calculations_data_frame.columns:
            if not self.calculations_data_frame[col].isna().sum(): continue
            imputer = Interpolate(self.calculations_data_frame[col], propagate_log = False)
            imputer.impute()
            self.calculations_data_frame[col] = imputer.get_imputed_values()
        
        self.calculate_imputation_errors()
        self.imputation_data_dict['Interpolation Class'] = self.error_data_frame.to_dict()

    def knn_method(self):
        """
        Perform k-nearest neighbors imputation for missing values in the dataframe.
        """
        from knn import KNN
        imputer = KNN(k = 5)
        self.calculations_data_frame = self.get_test_data_frame()
        for col in self.calculations_data_frame.columns:
            if not self.calculations_data_frame[col].isna().sum(): continue
            imputer.impute(self.test_gaps[col], col)
            self.calculations_data_frame[col].loc[self.test_gaps[col]] = imputer.get_imputed_values()
        
        self.calculate_imputation_errors()
        self.imputation_data_dict['KNN'] = self.error_data_frame.to_dict()

    def rnn_method(self):
        """
        Perform recurrent neural network imputation for missing values in the dataframe.
        """
        from rnn import RNN
        self.calculations_data_frame = self.get_test_data_frame()
        for col in self.calculations_data_frame.columns:
            if not self.calculations_data_frame[col].isna().sum(): continue
            gap_length = len(self.test_gaps[col])
            first_nan_index = self.calculations_data_frame[[col]][self.calculations_data_frame[col].isna()].index[0]
            X = self.calculations_data_frame[col].loc[first_nan_index - pd.Timedelta(800, 'S'):first_nan_index][:-1]
            imputer = RNN(variable_name = col)
            imputer.load_model()
            imputed_values = imputer.predict(X)[0]
            self.calculations_data_frame[col].loc[self.test_gaps[col]] = imputed_values[:gap_length]
        
        self.calculate_imputation_errors()
        self.imputation_data_dict['RNN'] = self.error_data_frame.to_dict()

    def arima_method(self):
        """
        Perform autoregressive integrated moving average (ARIMA) imputation for missing values in the dataframe.
        """
        from arima import ARIMA
        self.calculations_data_frame = self.get_test_data_frame()
        for col in self.calculations_data_frame.columns:
            if not self.calculations_data_frame[col].isna().sum(): continue
            first_nan_index = self.calculations_data_frame[[col]][self.calculations_data_frame[col].isna()].index[0]
            X = np.array(self.calculations_data_frame[col].loc[:first_nan_index])[:-1]
            p = 2 * len(self.test_gaps[col])
            q = 1 * len(self.test_gaps[col])
            imputer = ARIMA(p=p,q=q)
            imputer.fit(X)
            imputed_values = imputer.forecast(steps = len(self.test_gaps[col]))
            self.calculations_data_frame[col].loc[self.test_gaps[col]] = imputed_values
        
        self.calculate_imputation_errors()
        self.imputation_data_dict['ARIMA'] = self.error_data_frame.to_dict()

    
    # ====================================================================== #
    # ========================= Imputation Calc ============================ #
    # ====================================================================== #

    def calculate_imputation_errors(self):
        '''
        Calculates imputation errors between the imputed dataframe and the original dataframe for the specified test gaps.
        
        Args:
            imputed_df (pandas.DataFrame): The imputed dataframe.
            copy_df (pandas.DataFrame): A copy of the original dataframe.
            test_gaps (dict): A dictionary containing the test gaps for each column in the dataframe.
            
        Returns:
            pandas.DataFrame: A dataframe containing the calculated imputation errors (MAE, MSE, RMSE, R2) for each column.
        '''
        self.error_data_frame = {}
        for col in self.calculations_data_frame.columns:
            pred_val = self.calculations_data_frame[col].loc[self.test_gaps[col]]
            test_val = self.imputation_data_frame[col].loc[self.test_gaps[col]]
            mae = ImputationModules.mae(test_val,pred_val)
            mse = ImputationModules.mse(test_val, pred_val)
            rmse = ImputationModules.rmse(test_val, pred_val)
            r2 = ImputationModules.r2_score(test_val, pred_val)
            self.error_data_frame[col] = [mae,mse,rmse,r2]
            
        self.error_data_frame = pd.DataFrame(self.error_data_frame).T
        self.error_data_frame.columns = ['mae','mse','rmse','r2']

    @staticmethod
    def mae(real_values, predicted_values):
        """
        Calculate the mean absolute error (MAE) between real and predicted values.

        Args:
            real_values (numpy.ndarray): Array of real values.
            predicted_values (numpy.ndarray): Array of predicted values.

        Returns:
            float: The mean absolute error.
        """
        return sum((abs(real - predicted) for real, predicted in zip(real_values, predicted_values))) / len(real_values)
    
    @staticmethod
    def mse(real_values, predicted_values):
        """
        Calculate the mean squared error (MSE) between real and predicted values.

        Args:
            real_values (numpy.ndarray): Array of real values.
            predicted_values (numpy.ndarray): Array of predicted values.

        Returns:
            float: The mean squared error.
        """
        return sum(((real - predicted)**2 for real, predicted in zip(real_values, predicted_values))) / len(real_values)

    @staticmethod
    def rmse(real_values, predicted_values):
        """
        Calculate the root mean squared error (RMSE) between real and predicted values.

        Args:
            real_values (numpy.ndarray): Array of real values.
            predicted_values (numpy.ndarray): Array of predicted values.

        Returns:
            float: The root mean squared error.
        """
        return ImputationModules.mse(real_values, predicted_values) ** 0.5

    @staticmethod
    def r2_score(real_values, predicted_values):
        """
        Calculate the R-squared score between real and predicted values.

        Args:
            real_values (numpy.ndarray): Array of real values.
            predicted_values (numpy.ndarray): Array of predicted values.

        Returns:
            float: The R-squared score.
        """
        mean_value = sum(real_values) / len(real_values)
        sum_of_squares_total = sum((real_value - mean_value) ** 2 for real_value in real_values)
        if sum_of_squares_total == 0: return -1
        sum_of_squares_residuals = sum((real_value - predicted_value) ** 2 for real_value, predicted_value in zip(real_values, predicted_values))
        r_score = 1 - (sum_of_squares_residuals / sum_of_squares_total)
        return r_score

    # ====================================================================== #
    # ======================== Getters & Setters =========================== #
    # ====================================================================== #

    def get_test_gaps(self):
        """
        Get the dictionary of test gaps.

        Returns:
            dict: A dictionary where keys are column names and values are lists of indexes 
                representing NaN gaps in the corresponding columns for the specified month.
        """
        return dict(self.test_gaps)
    
    def get_test_data_frame(self):
        """
        Get a copy of the test imputation data frame.

        Returns:
            pandas.DataFrame: A copy of the test imputation data frame.
        """
        return self.test_imputation_data_frame.copy()
    
    def get_imputation_data_frame(self):
        """
        Get a copy of the imputation data frame.

        Returns:
            pandas.DataFrame: A copy of the imputation data frame.
        """
        return self.imputation_data_frame.copy()
    
    def get_meteo_data_frame(self):
        """
        Get a copy of the Meteoblue data frame.

        Returns:
            pandas.DataFrame: A copy of the Meteoblue data frame.
        """
        return self.meteoblue_data_frame.copy()
    
    def get_imputation_methods_data(self):
        """
        Get a dataframe containing imputation methods data.

        Returns:
            pandas.DataFrame: A dataframe containing imputation methods data.
        """
        return pd.DataFrame.from_dict(
            {(outerKey, innerKey): values for outerKey, innerDict in self.imputation_data_dict.items()
             for innerKey, values in innerDict.items()}
             ).T
    
    def set_path(self, path):
        """
        Set the path attribute.

        Args:
            path (str): The path to set.
        """
        self.path = path
    