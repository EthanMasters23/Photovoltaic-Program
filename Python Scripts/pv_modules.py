#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 12 16:03 2023

@author: Ethan Masters

Description: PV Project Modules Class designed to modularize 
    code for cross program use.

Python Version: Python 3.10.11 (main, Apr 20 2023, 13:59:00)
"""

import os
import pandas as pd
import numpy as np
import pvlib
import re
import logging
import time
from pv_constants import COL_NAMES


class PVModules:
    """
    A class to process photovoltaic (PV) module data.

    Attributes:
        - FILE_TYPE (str): Type of file to process.
        - YEAR (str): Year of the data to process.
        - MONTH (str): Month of the data to process.
        - data_frame (pandas.DataFrame): DataFrame containing PV module data.
        - propagate_logger (bool): Flag to propagate logging messages.
        - path_list (list): List of file paths matching the given FILE_TYPE, YEAR, and MONTH.
        - clean_path_list (list): List of cleaned file paths.
        - pv_logger (logging.Logger): Logger for PV module operations.

    """

    def __init__(self, FILE_TYPE = None, YEAR = None, MONTH = None, data_frame = pd.DataFrame(), propagate_logger = True):
        """
        Initializes PVModules class with provided parameters.

        Args:
            - FILE_TYPE (str): Type of file to process.
            - YEAR (str): Year of the data to process.
            - MONTH (str): Month of the data to process.
            - data_frame (pandas DataFrame): DataFrame containing PV module data.
            - propagate_logger (bool): Flag to propagate logging messages.

        Raises:
            - Exception: If incorrect input parameters are provided.

        """
        if FILE_TYPE != None and not [file_i for file_i in ['Irradiance','Deger','Fixed'] if re.search(fr'{FILE_TYPE}',file_i)]:
            raise Exception(f"Incorret Input: File")
        if YEAR != None and not re.search(r'\d{4}',YEAR):
            raise Exception(f"Incorret Input: Year")
        if MONTH != None and not re.search(r'[A-Za-z]{3}',MONTH):
            print("raised exception")
            raise Exception(f"Incorret Input: Month")

        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH
        self.data_frame = data_frame

        self.path_list = []
        self.clean_path_list = []

        (self.create_file_path()
        if self.FILE_TYPE
        else None)
        (self.create_path_function() 
        if self.YEAR and self.MONTH
        else None)

        self.pv_logger = logging.getLogger(type(self).__name__)
        self.pv_logger.propagate = propagate_logger

    def create_file_path(self):
        """
        Retrieves the file paths for a given file name from the data directory.

        Returns:
            - list: List of file paths matching the given FILE_TYPE.

        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        for dir in os.scandir(file_path):
            if re.search(r'\.',dir.name): continue
            year_path = os.path.join(file_path , f"{dir.name}")
            for dir in os.scandir(year_path):
                if dir.name == self.FILE_TYPE:
                    month_path = os.path.join(year_path , f"{dir.name}")
                    for dir in os.scandir(month_path):
                        if re.search(r'\.csv|\.xlsx',dir.name):
                            if re.search(r'clean|test',dir.name):
                                self.clean_path_list += [os.path.join(month_path , f"{dir.name}")]
                            else:
                                self.path_list += [os.path.join(month_path , f"{dir.name}")]

    def create_path_function(self):
        """
        Searches for a specific path within the provided path_list based on the given year and month parameters.

        Returns:
            - list: A list containing the path(s) that match the specified year and month.

        """
        for path in self.path_list:
            y, m = re.search(r"/(\d{4})/[a-z]*/([a-z]*)\.", path.lower()).group(1, 2)
            if re.search(fr"{self.YEAR}", y) and re.search(fr"{self.MONTH}", m):
                self.path_list = [path]
                break

    def reshape_df(self):
        """
        Reshapes the input DataFrame by manipulating the 'DayID' and 'TimeID' columns.

        Returns:
            - pandas.DataFrame: Reshaped DataFrame with the 'date' column as the index.

        """
        self.pv_logger.info("Reshaping data frame using .reshape().")
        self.data_frame['DayID'] = self.data_frame['DayID'].astype(str)
        self.data_frame['TimeID'] = self.data_frame['TimeID'].astype(str)
        self.data_frame['index'] = self.data_frame['DayID'] + 'T' + self.data_frame['TimeID']
        self.data_frame = self.data_frame.drop(columns=['DayID', 'TimeID'])
        self.data_frame['index'] = pd.to_datetime(self.data_frame['index'])
        self.data_frame = self.data_frame.set_index('index')
        self.data_frame.index = self.data_frame.index.tz_localize(tz='Etc/UTC')
        self.data_frame = self.data_frame.sort_index()

        month = self.data_frame.index[0].month
        year = self.data_frame.index[0].year

        if self.FILE_TYPE == 'Fixed' and month in [10,11,12] and year == 2020:
            self.data_frame.columns = ['MonoSi_Vin','MonoSi_Iin','MonoSi_Vout','MonoSi_Iout','PolySi_Vin','PolySi_Iin','PolySi_Vout','PolySi_Iout','TFSi_a_Vin','TFSi_a_Iin','TFSi_a_Vout','TFSi_a_Iout','TempF_Mono','TempF_Poly','TempF_Amor','TempF_Cigs']
        else:
            self.data_frame.columns = COL_NAMES[self.FILE_TYPE]
        
        self.data_frame = self.data_frame.apply(pd.to_numeric)


    def resample_df(self):
        """
        Resamples the DataFrame to have a 20-second frequency and checks for missing timestamps edge cases.

        Returns:
            - pandas.DataFrame: DataFrame with resampled data and sorted index.

        """
        frequency = '20s'
        self.pv_logger.info(f"Resampling data frame using resample_df() to {frequency}.")
        
        # creating of list of times to find interval gaps
        time_list = self.data_frame.index
        interval_list = []

        # checking for missing values at the beginning of the month
        first_observation = time_list[0]
        if first_observation > first_observation.replace(day = 1, hour = 0, minute = 0, second = 30):
            interval_list += list(pd.date_range(start = first_observation.replace(day = 1, hour = 0, minute = 0, second = 0),
                                                end = first_observation,
                                                freq = frequency))
        
        # checking for missing values at the end of the month    
        last_observation = time_list[-1]
        next_month = last_observation.replace(day = 28,hour = 0, minute = 0, second = 0) + pd.Timedelta(4, 'd')
        last_day = next_month - pd.Timedelta(next_month.day, 'd')
        if last_observation < last_day.replace(hour = 23, minute = 59, second = 30):
            interval_list += list(pd.date_range(start = last_observation,
                          end = last_day.replace(hour = 23, minute = 59, second = 59),
                          freq = frequency))
            
        if interval_list:
            missing_times_data_frame = pd.DataFrame(index = interval_list, columns = self.data_frame.columns)
            missing_times_data_frame.loc[:, :] = np.nan
            self.data_frame = pd.concat([self.data_frame, missing_times_data_frame], axis = 0).sort_index()

        self.data_frame = self.data_frame.resample(frequency).mean()

    def clean_irradiance_values(self):
        """
        Cleans irradiance values in a DataFrame by removing outliers and negative values.

        Returns:
            - pandas.DataFrame: DataFrame with cleaned irradiance values.

        """
        self.pv_logger.info("Cleaning data frame values using .clean_irradiance_values().")
        # Removing DirectIR Values #
        self.data_frame[self.data_frame['GlobalIR'] > 2200] = np.nan
        # Removing Temperature Values #
        self.data_frame[self.data_frame['Temperature'] > 50] = np.nan
        # Removing Wind Speed Values #
        self.data_frame['WindSpeed'] = abs(self.data_frame['WindSpeed'])
        self.data_frame[self.data_frame['WindSpeed'] > 10] = np.nan
        # Removing DirectIR Values #
        self.data_frame[self.data_frame['DirectIR'] > 1000] = np.nan
        # Removing DiffuseIR Values #
        self.data_frame[self.data_frame['DiffuseIR'] > 1200] = np.nan
        # Removing Negative Values #
        col_list = [col for col in self.data_frame.columns if not re.search("temp", col.lower())]
        self.data_frame[(self.data_frame < 0) & (self.data_frame.columns.isin(col_list))] = np.nan

    def clean_deger_fixed_values(self):
        """
        Cleans Deger & Fixed values in a DataFrame by removing negative values.

        Returns:
            - pandas.DataFrame: DataFrame with cleaned values.

        """
        self.pv_logger.info("Cleaning data frame values using .clean_deger_fixed_values().")
        self.data_frame[self.data_frame < 0] = np.nan
    
    def time_features(self):
        self.data_frame = self.add_time_features(self.data_frame)

    @classmethod
    def add_time_features(cls, data_frame):
        """
        Extracts day, month, and year features from the index of a DataFrame.

        Returns:
            - pandas.DataFrame: Modified DataFrame with 'day', 'month', and 'year' columns added.

        """
        data_frame['day'] = [d.day for d in data_frame.index]
        data_frame['month'] = [d.month for d in data_frame.index]
        data_frame['year'] = [d.year for d in data_frame.index]
        return data_frame

    def remove_night(self):
        self.pv_logger.info("Removing night time values using .remove_night().")
        self.data_frame = self.remove_nighttime_values(self.data_frame)

    @classmethod
    def remove_nighttime_values(cls, data_frame):
        """
        Removes nighttime data from a DataFrame based on solar position information.

        Returns:
            - pandas.DataFrame: DataFrame with nighttime data removed.

        """
        lat = 49.102
        lon = 6.215
        alt = 220
        solpos = pvlib.solarposition.get_solarposition(
            time = data_frame.index, latitude = lat, longitude = lon, altitude = alt, method = 'pyephem')
        return  data_frame[solpos['zenith'] <= 90]
    
    def set_night_values(self, night_value = 0):
        self.pv_logger.info("Setting night time data frame values to zero using .set_night_values().")
        self.data_frame = self.set_nighttime_values(self.data_frame, night_value = night_value)

    @classmethod
    def set_nighttime_values(cls, data_frame, night_value = 0):
        """
        Sets nighttime data frame values to zero.

        Returns:
            - pandas.DataFrame: DataFrame with nighttime values set to zero.

        """
        lat = 49.102
        lon = 6.215
        alt = 220
        solpos = pvlib.solarposition.get_solarposition(
            time = data_frame.index, latitude = lat, longitude = lon, altitude = alt, method = 'pyephem')
        col_list = [col for col in data_frame.columns if re.search("ir", col.lower())]
        data_frame.loc[solpos['zenith'] > 90, data_frame.columns.isin(col_list)] = night_value
        return data_frame
    
    @staticmethod
    def time_converter(seconds):
        """
        Converts seconds to hours, minutes, and seconds.

        Args:
            - seconds (int): Number of seconds to convert.

        Returns:
            - str: Converted time string in HH:MM:SS format.

        """
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    
    @staticmethod
    def start_logger(file_name):
        """
        Starts logging operations.

        Args:
            - file_name (str): Name of the log file.

        Returns:
            - logging.Logger: Logger object for logging operations.

        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', f'{file_name}_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        return logger
    
    @staticmethod
    def stop_logger(logger, start_time):
        """
        Stops logging operations and calculates total runtime.

        Args:
            - logger (logging.Logger): Logger object for logging operations.
            - start_time (float): Start time of the process.

        """
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")


    # ====================================================================== #
    # ======================== Getters & Setters =========================== #
    # ====================================================================== #

    def get_data_frame(self):
        """
        Gets the DataFrame containing PV module data.

        Returns:
            - pandas.DataFrame: DataFrame containing PV module data.

        """
        return self.data_frame

    def get_path_list(self):
        """
        Gets the list of file paths.

        Returns:
            - list: List of file paths.

        """
        return self.path_list
    
    def get_file_type(self):
        """
        Gets the type of file to process.

        Returns:
            - str: Type of file to process.

        """
        return self.FILE_TYPE

    def set_data_frame(self, data_frame):
        """
        Sets the DataFrame containing PV module data.

        Args:
            - data_frame (pandas.DataFrame): DataFrame containing PV module data.

        """
        self.data_frame = data_frame