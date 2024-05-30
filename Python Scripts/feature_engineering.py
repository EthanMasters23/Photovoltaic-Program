#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on June 2 10:03 2023

@author: Ethan Masters

Purpose: Feature Engineering Script, used to develop
    and save features for imputation methods.

Python Version: Python 3.10.11 
"""

import pandas as pd
import numpy as np
import logging
import os
import pvlib
import time
from sklego.preprocessing import RepeatingBasisFunction
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pv_modules import PVModules
import re
import math


class FeatureEngineering:
    def __init__(self, input_data_frame = None, propagate_log = True):
        self.SECONDS_COL = ['Morning', 'Noon', 'Evening', 'Night']
        self.DAY_COL = ['Winter', 'Spring', 'Summer', 'Fall']
        self.TIME_PERIODS = {
            'D': 'day',
            'W': 'week',
            'H': 'hour',
            '2H': '2 hour',
            '4H': '4 hour',
            '8H': '8 hour',
            '12H': '12 hour',
            '15T': '15 minute',
            '30T': '30 minute'
        }
        self.feature_data_frame = (pd.DataFrame(index = input_data_frame.index) if type(input_data_frame) == pd.DataFrame else pd.DataFrame())
        self.training_data_frame = (input_data_frame.copy() if type(input_data_frame) == pd.DataFrame else pd.DataFrame())
        self.non_nan_index = (input_data_frame.dropna().index if type(input_data_frame) == pd.DataFrame else pd.DataFrame())
        self.clean_training_data = {}
        self.feature_logger = logging.getLogger(type(self).__name__)
        self.feature_logger.propagate = propagate_log

    def remove_nan_rows(self):
        self.feature_data_frame = self.feature_data_frame.loc[self.non_nan_index]
        self.training_data_frame = self.training_data_frame.loc[self.non_nan_index]
        self.feature_logger.info("Removed NaN rows.")

    def remove_night(self):
        self.feature_data_frame = PVModules.remove_nighttime_values(self.feature_data_frame)
        self.training_data_frame = PVModules.remove_nighttime_values(self.training_data_frame)
        self.feature_logger.info("Removed night values.")

    def set_night(self):
        self.feature_data_frame = PVModules.set_nighttime_values(self.feature_data_frame)
        self.training_data_frame = PVModules.set_nighttime_values(self.training_data_frame)
        self.feature_logger.info("Set night values to 0.")

    def generate_time_features_rbf(self):
        """
        Performs feature engineering on a given DataFrame by adding time-based features using radial basis functions.
        
        The function calculates two additional time-based features: 'seconds_of_day' and 'seconds_of_year'. 
        These features capture the number of seconds elapsed within a day and within a year, respectively, for each timestamp in the index of the input DataFrame.
        Radial basis functions are then applied to these features to create transformed features.
        
        Args:
            feature_df (DataFrame): The input DataFrame containing timestamps in the index.
        
        Returns:
            DataFrame: The updated DataFrame with added time-based features.
        """
        self.feature_data_frame['seconds_of_day'] = [(time.replace(hour = 0, minute = 0, second = 0, microsecond = 0) - time).total_seconds() for time in self.feature_data_frame.index]
        self.feature_data_frame['seconds_of_year'] = self.feature_data_frame.index.day_of_year
        rbf_day_of_year = RepeatingBasisFunction(n_periods = 4, column = "seconds_of_year", remainder = "drop")
        rbf_day_of_year.fit(self.feature_data_frame)
        day_of_year_features = pd.DataFrame(index = self.feature_data_frame.index, columns = self.DAY_COL, data = rbf_day_of_year.transform(self.feature_data_frame))
        rbf_second = RepeatingBasisFunction(n_periods = 4, column = "seconds_of_day", remainder = "drop")
        rbf_second.fit(self.feature_data_frame)
        seconds_feature = pd.DataFrame(index = self.feature_data_frame.index, columns = self.SECONDS_COL, data = rbf_second.transform(self.feature_data_frame))
        self.feature_data_frame = self.feature_data_frame.drop(['seconds_of_day', 'seconds_of_year'], axis = 1)
        self.feature_data_frame = pd.concat([self.feature_data_frame, seconds_feature, day_of_year_features], axis = 1, ignore_index = False)

    def generate_time_features_cos_sin_trans(self):
        """
        Generates time-based features using sine and cosine functions for the seconds of the day.

        Args:
            feature_df (DataFrame): The input DataFrame containing timestamps in the index.

        Returns:
            DataFrame: The updated DataFrame with added time-based features.
        """
        seconds_of_day = (self.feature_data_frame.index.hour * 3600) + (self.feature_data_frame.index.minute * 60) + self.feature_data_frame.index.second
        day_of_year = self.feature_data_frame.index.dayofyear
        sin_seconds_of_day = np.sin(2 * np.pi * seconds_of_day / 86400)
        cos_seconds_of_day = np.cos(2 * np.pi * seconds_of_day / 86400)
        sin_day_of_year = np.sin(2 * np.pi * day_of_year / 365)
        cos_day_of_year = np.cos(2 * np.pi * day_of_year / 365)
        time_features_df = pd.DataFrame(index=self.feature_data_frame.index)
        time_features_df['sin_seconds_of_day'] = sin_seconds_of_day
        time_features_df['cos_seconds_of_day'] = cos_seconds_of_day
        time_features_df['sin_day_of_year'] = sin_day_of_year
        time_features_df['cos_day_of_year'] = cos_day_of_year
        self.feature_data_frame = pd.concat([self.feature_data_frame, time_features_df], axis=1)
        self.feature_logger.info(f"Generated time features with sin and cos transformation functions.")

    def generate_solar_zenith_feature(self):
        """
        Adds a column for the solar zenith angle to the input data.

        Parameters:
        - data: The input DataFrame with a datetime index.
        - location: A pvlib Location object representing the location of interest.

        Returns:
        - The input DataFrame with an additional column for the solar zenith angle.
        """
        lat = 49.102
        lon = 6.215
        alt = 220
        solpos = pvlib.solarposition.get_solarposition(
            time = self.feature_data_frame.index,
            latitude = lat,
            longitude = lon,
            altitude = alt,
            method = 'pyephem')
        self.feature_data_frame['Solar Position'] = solpos['zenith']
        self.feature_logger.info("Generated Solar Position Feature.")

    def generate_clean_data(self, period):
        self.feature_logger.info("------------------ generating clean data series for model training ------------------")
        midnight = pd.Timestamp('2022-01-01 00:00:00').tz_localize(tz='Etc/UTC').time()
        sunrise_time = pd.Timestamp('2022-01-01 05:43:22').tz_localize(tz='Etc/UTC').time()
        sunset_time = pd.Timestamp('2022-01-01 17:26:31').tz_localize(tz='Etc/UTC').time()
        periods = self.training_data_frame.index.to_period(period).unique()
        self.feature_logger.info(f"Number of {self.TIME_PERIODS[period]}s ({period}) in training data: {len(periods)}")
        self.clean_training_data = {}
        self.training_data_frame = PVModules.set_night_values(self.training_data_frame, night_value = np.nan)
        for col in tqdm(self.training_data_frame.columns):
            self.clean_training_data[col] = []
            for p in periods:
                start_time = p.start_time.tz_localize(tz='Etc/UTC')
                end_time = p.end_time.tz_localize(tz='Etc/UTC')
                if (
                    (start_time.month == 7 and start_time.year == 2022 and end_time.month == 7 and end_time.year == 2022)
                ): continue
                clean_period = self.training_data_frame[col].loc[start_time:end_time]
                if (
                    (len(clean_period) != math.ceil((end_time - start_time).total_seconds() / 20))
                    or (math.ceil((end_time - start_time).total_seconds() / 20) != 45 and len(clean_period) != 45)
                    or (clean_period.isna().sum())
                ): continue
                clean_period_features = self.feature_data_frame.loc[start_time:end_time] if not self.feature_data_frame.empty else []
                self.clean_training_data[col] += [list(clean_period.values) + ([list(aArray) for aArray in clean_period_features.values] if clean_period_features else [])]

            if not self.clean_training_data[col]:
                self.feature_logger.info(f"There are no clean {self.TIME_PERIODS[period]}s ({period}) for {col}.")
            else:
                self.feature_logger.info(f"Number of clean {self.TIME_PERIODS[period]}s ({period}) for {col}: {len(self.clean_training_data[col])}")
                self.feature_logger.info(f"Percentage of {self.TIME_PERIODS[period]} series in total {self.TIME_PERIODS[period]}s in {self.TIME_PERIODS[period]}_training_data for {col}: {round((len(self.clean_training_data[col]) / len(periods)) * 100, 2)}%")

            # - Uncomment to not return full dataframe and save - #
            # - as running (decreased memory inc time complexity) - #
            self.save_clean_training_data(period)
            del self.clean_training_data[col]

        self.feature_logger.info(f"Finished generating series of clean {self.TIME_PERIODS[period]}s ({period}) for model training.")

    def graph_time_features_rbf(self):
        day_of_year = self.feature_data_frame[self.DAY_COL]
        day_of_year.plot(subplots=True,
                        sharex=True,
                        title = "Day of Year Feature Engineering (Radial Basis Functions)",
                        legend=True)
        plt.grid(color='lightgray', linestyle='dashed')
        plt.tight_layout()
        plt.show()
        seconds_of_day = self.feature_data_frame[self.SECONDS_COL]
        seconds_of_day = seconds_of_day.loc[self.feature_data_frame.index[-1] - pd.Timedelta(1, 'day') : self.feature_data_frame.index[-1], :]
        seconds_of_day.plot(subplots=True,
                            sharex=True,
                            title="Seconds Feature Engineering (Radial Basis Function)",
                            legend=True)
        plt.grid(color='lightgray', linestyle='dashed')
        plt.tight_layout()
        plt.show()

    def graph_time_features_cos_sin_trans(self):
        seconds_of_day = self.feature_data_frame[['sin_seconds_of_day','cos_seconds_of_day']]
        seconds_of_day = seconds_of_day.loc[self.feature_data_frame.index[-1] - pd.Timedelta(1, 'day') : self.feature_data_frame.index[-1], :]
        seconds_of_day.plot(subplots=True, sharex=True, title="Seconds Feature Engineering (Cos & Sin)", legend=True)
        plt.grid(color='lightgray', linestyle='dashed')
        plt.tight_layout()
        plt.show()
        day_of_year = self.feature_data_frame[['sin_day_of_year', 'cos_day_of_year']]
        day_of_year.plot(subplots=True, sharex=True, title="Day of Year Feature Engineering (Cos & Sin)", legend=True)
        plt.grid(color='lightgray', linestyle='dashed')
        plt.tight_layout()
        plt.show()
        
    def save_features_data_frame(self):
        file = 'feature_data_frame.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file)
        self.feature_data_frame.to_csv(file_path)
        self.feature_logger.info(f"Saved feature data frame as '{file}'.")

    def save_training_data_frame(self):
        file = 'training_data_frame.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file)
        self.training_data_frame.to_csv(file_path)
        self.feature_logger.info(f"Saved training data frame as '{file}'.")

    def save_clean_training_data(self, period):
        for var in self.clean_training_data:
            file = f'{var}_training_data_{period.lower()}.json'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file)
            with open(file_path, 'w') as f:
                json.dump(self.clean_training_data[var], f)
        self.feature_logger.info(f"Saved clean {self.TIME_PERIODS[period]}s ({period}) data frame.")

    def load_features_data_frame(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'feature_data_frame.csv')
        self.feature_data_frame = pd.read_csv(file_path, index_col = 0)
        self.feature_data_frame.index = pd.to_datetime(self.feature_data_frame.index)

    def load_training_data_frame(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'training_data_frame.csv')
        self.training_data_frame = pd.read_csv(file_path, index_col = 0)
        self.training_data_frame.index = pd.to_datetime(self.training_data_frame.index)

    def load_clean_training_data(self, period):
        file = f'training_data_{period.lower()}.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file)
        with open(file_path, 'r') as f:
            self.clean_training_data = json.load(f)

    def get_clean_training_data(self):
        return self.clean_training_data

    def get_features_data_frame(self):
        return self.feature_data_frame
    
    def get_solar_position_feature(self):
        return self.feature_data_frame[['Solar Position']]
    
    def get_time_features(self):
        return self.feature_data_frame[self.SECONDS_COL + self.DAY_COL]
    
    def get_day_features(self):
        return self.feature_data_frame[self.SECONDS_COL]

    def get_season_feature(self):
        return self.feature_data_frame[self.DAY_COL]
    
    def get_training_data(self):
        return self.training_data_frame
    
class FeatureEngineeringPipeline:
    def __init__(self, FILE_TYPE):
        self.FILE_TYPE = FILE_TYPE
        self.run()

    def run(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', f'feature_engineering_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.logger.info(f"Starting feature engineering...")
        file = f'{self.FILE_TYPE}_compiled_processed.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file)
        data_frame = pd.read_csv(file_path, index_col = 0)
        data_frame.index = pd.to_datetime(data_frame.index)
        data_frame = data_frame.sort_index()
        self.logger.info(f"Engineering features using '{file}' ({data_frame.index[0]} - {data_frame.index[-1]}).")
        self.feature_engineer = FeatureEngineering(
            input_data_frame = data_frame
        )

    def run_main(self):
        # = Method for creating training data = #
        # feature_engineer.generate_solar_zenith_feature()
        # feature_engineer.generate_time_features_cos_sin_trans()
        # feature_engineer.remove_nan_rows()
        # feature_engineer.remove_night()
        # feature_engineer.save_features_data_frame()
        # feature_engineer.save_training_data_frame()
        # feature_engineer.load_features_data_frame()
        # feature_engineer.load_training_data_frame()

        # = Method for creating sequential training data = #
        period = '15T'
        # feature_engineer.generate_time_features_cos_sin_trans()
        # feature_engineer.set_night()
        self.feature_engineer.generate_clean_data(period)
        end_time = time.time()
        total_time = end_time - self.start_time
        self.logger.info("Complete.")
        self.logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        self.logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")
    


if __name__ == "__main__":
    FeatureEngineeringPipeline(
        FILE_TYPE = 'Irradiance'
    ).run_main()