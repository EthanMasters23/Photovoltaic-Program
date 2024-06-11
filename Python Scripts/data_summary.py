#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on March 18 10:23 2023

@author: Ethan Masters

Purpose: Data Summary Class, for summarizing data by
    variable for general purposes. Ouputs are stored
    in the logs/ directory.

Python Version: Python 3.10.11 
"""

import pandas as pd
import logging
import plotly.express as px
import numpy as np
import os
import time
import pvlib
from scipy.stats import spearmanr
import datetime
from data_compiler import DataCompiler
from pv_modules import PVModules


class DataSummary:
    """
    A class to summarize data for a specific file type, year, and month.

    Attributes:
        FILE_TYPE (str): Type of data to summarize.
        YEAR (int): Year for summarizing the data.
        MONTH (int): Month for summarizing the data.
        data_frame (pandas DataFrame): DataFrame containing summarized data.
        data_frame_raw (pandas DataFrame): Raw DataFrame containing original data.
        summary_logger (logging.Logger): Logger for data summarization operations.

    """

    def __init__(self, FILE_TYPE = None, YEAR = None, MONTH = None, propagagte_logger = True):
        """
        Initializes DataSummary class with provided parameters.

        Args:
            FILE_TYPE (str): Type of data to summarize.
            YEAR (int): Year for summarizing the data.
            MONTH (int): Month for summarizing the data.
            propagate_logger (bool): Flag to propagate logging messages.

        """
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH
        self.data_frame = pd.DataFrame()
        self.data_frame_raw = pd.DataFrame()
        self.summary_logger = logging.getLogger(type(self).__name__)
        self.summary_logger.propagate = propagagte_logger

    def run_data_summary(self):
        """
        Runs the data summarization process.

        This method summarizes the data based on the provided parameters (FILE_TYPE, YEAR, MONTH).

        """
        self.summary_logger.info(f"Running {self.FILE_TYPE} summary for all available data.")
        if self.YEAR and self.MONTH:
            pv_module = PVModules(
                FILE_TYPE=self.FILE_TYPE,
                YEAR = self.YEAR,
                MONTH = self.MONTH
            )
            path_list = pv_module.get_path_list()
            df = pd.read_csv(path_list[0], sep = "\t|,", engine = 'python')
            pv_module.set_data_frame(df)
            pv_module.reshape_df()
            self.data_frame_raw = pv_module.get_data_frame().copy()
            if self.FILE_TYPE == "Irradiance":
                pv_module.clean_irradiance_values()
            else:
                pv_module.clean_deger_fixed_values()
            pv_module.set_night_values()
            pv_module.resample_df()
            self.data_frame = pv_module.get_data_frame()
            return

        data_compiler_processed = DataCompiler(FILE_TYPE=self.FILE_TYPE, FILE_OPTION='processed')
        data_compiler_processed.run()
        data_compiler_processed.save_compiled_data()
        data_compiler_raw = DataCompiler(FILE_TYPE=self.FILE_TYPE, FILE_OPTION='raw')
        data_compiler_raw.run()
        data_compiler_raw.save_compiled_data()
        self.data_frame = data_compiler_processed.get_compiled_data()
        self.data_frame_raw = data_compiler_raw.get_compiled_data()

    def load_data_summary(self):
        """
        Loads previously summarized data.

        This method loads previously compiled data for the current summary of all available data.

        """
        self.summary_logger.info(f"Loading previously compiled {self.FILE_TYPE} data for current summary of all available data.")
        data_compiler_processed = DataCompiler(FILE_TYPE = self.FILE_TYPE, FILE_OPTION = 'processed')
        data_compiler_processed.load_compiled_data()
        data_compiler_raw = DataCompiler(FILE_TYPE = self.FILE_TYPE, FILE_OPTION = 'raw')
        data_compiler_raw.load_compiled_data()
        self.data_frame = data_compiler_processed.get_compiled_data()
        self.data_frame_raw = data_compiler_raw.get_compiled_data()

    def summarize_data(self):
        self.observation_times_summary()
        self.summary_by_variable()
        self.average_peak_time_and_value()
        self.average_sunrise_sunset_time()
        self.outliers(pre = True)
        self.outliers(pre = False)
        self.corr_matrix()
        self.plot_daily_seasonality()
        self.visualize_seasonality()

    def outliers(self, pre):
        """
        Summarizes outliers in the data.

        Args:
            pre (bool): Flag indicating whether to summarize outliers before or after preprocessing.

        """
        (self.summary_logger.info("------------------------ Summary of outliers pre-processing ------------------------")
         if pre else self.summary_logger.info("------------------------ Summary of outliers post-processing ------------------------"))
        outlier_indicator = True
        if pre: 
            df = self.data_frame.copy()
        else:
            df = self.data_frame_raw.copy()
        for col in df.columns:
            arr = df[col]
            z_scores = np.abs((arr - arr.mean()) / arr.std())
            threshold = 3
            outliers = arr[z_scores > threshold]
            if len(outliers):
                outlier_indicator = False
                self.summary_logger.info(f"{col} number of outliers {len(outliers)}, min: {sorted(outliers)[0]}, max: {sorted(outliers)[-1]}")
        if outlier_indicator:
            (self.summary_logger.info("There were no outliers found pre-processing.")
            if pre else self.summary_logger.info("There were no outliers found post-processing."))

    def average_peak_time_and_value(self):
        """
        Summarizes average peak values and times in the data.

        """
        self.summary_logger.info("------------------------ Average Peak Values / Time ------------------------")
        for col in self.data_frame.columns:
            daily_max_value = self.data_frame[[col]].resample('D').max().mean()
            daily_max_time = self.data_frame[[col]].groupby(self.data_frame.index.floor('D')).idxmax()
            max_time_timestamps = pd.to_datetime(daily_max_time[col])
            max_times = max_time_timestamps.dt.time
            max_times_seconds = max_times.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            average_time_seconds = max_times_seconds.mean()
            average_hours = int(average_time_seconds / 3600)
            average_minutes = int((average_time_seconds % 3600) / 60)
            average_seconds = int(average_time_seconds % 60)
            average_time = datetime.time(average_hours, average_minutes, average_seconds)
            self.summary_logger.info(f"Average Peak {col} Value in a Day: {round(daily_max_value[0], 2)}.")
            self.summary_logger.info(f"Average Peak {col} Time in a Day: {average_time}.")

    def average_sunrise_sunset_time(self):
        """
        Summarizes average sunrise and sunset times based on the data.

        """
        self.summary_logger.info("------------------------ Average Sunrise / Sunset Times ------------------------")
        lat = 49.102
        lon = 6.215
        alt = 220
        solpos = pvlib.solarposition.get_solarposition(
            time = self.data_frame.index, latitude = lat, longitude = lon, altitude = alt, method = 'pyephem')
        
        solpos = solpos[solpos['zenith'] <= 90]
        solpos['time'] = solpos.index
        sunrise_times = solpos.groupby(solpos.index.date).first()
        sunset_times = solpos.groupby(solpos.index.date).last()
        for data_frame in [sunrise_times, sunset_times]:
            max_time_timestamps = pd.to_datetime(data_frame['time'])
            max_times = max_time_timestamps.dt.time
            max_times_seconds = max_times.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            average_time_seconds = max_times_seconds.mean()
            average_hours = int(average_time_seconds / 3600)
            average_minutes = int((average_time_seconds % 3600) / 60)
            average_seconds = int(average_time_seconds % 60)
            average_time = datetime.time(average_hours, average_minutes, average_seconds)
            (self.summary_logger.info(f"Average sunrise time: {average_time}.")
             if data_frame is sunrise_times else self.summary_logger.info(f"Average sunset time: {average_time}."))
            
    def summary_by_variable(self):
        """
        Summarizes data variables.

        """
        self.summary_logger.info("------------------------ Variable Summary ------------------------")
        for col in self.data_frame.columns:
            df_description = self.data_frame[[col]].describe()
            self.summary_logger.info("\n" + str(df_description))

    def corr_matrix(self):
        """
        Generates a correlation matrix for the data.

        """
        self.summary_logger.info("------------------------ Correlation Matrix ------------------------")
        df = self.data_frame.copy()
        df = df.dropna()
        corrs = []
        p_values = []
        for feat1 in df.columns:
            corr_list = []
            p_list = []
            for feat2 in df.columns:
                corr, p_value = spearmanr(df[feat1], df[feat2])
                corr_list += [corr]
                p_list += [p_value]
            corrs += [corr_list]
            p_values += [p_list]
            
        corr_matrix = pd.DataFrame(corrs, index = df.columns, columns = df.columns)
        self.summary_logger.info("\n" + str(corr_matrix))
        px.imshow(corr_matrix, text_auto = True, title = "Correlation Matrix").show()

    def observation_times_summary(self):
        """
        Generates time observation dictionary at different intervals
        to visualize the gaps in time of missing observation intervals.

        """
        self.summary_logger.info("------------------------ Observation Times Summary Before Resampling ------------------------")
        index_list = self.data_frame_raw.index
        obs_times_dict = {}
        for index, time in enumerate(index_list):
            year = time.year
            month = time.month
            if index == len(index_list) - 1: continue
            time_lapsed = round((index_list[index + 1] - time).total_seconds())
            if time_lapsed <= 10:
                time_lapsed = f'(0 - 10)s'
            elif time_lapsed <= 20:
                time_lapsed = f'(10 - 20)s'
            elif time_lapsed <= 30:
                time_lapsed = f'(20 - 30)s'
            elif time_lapsed <= 40:
                time_lapsed = f'(30 - 40)s'
            elif time_lapsed <= 50:
                time_lapsed = f'(40 - 50)s'
            elif time_lapsed <= 60:
                time_lapsed = f'(50 - 60)s'
            elif time_lapsed <= 300:
                time_lapsed = f'(1 - 5)min'
            elif time_lapsed <= 600:
                time_lapsed = f'(5 - 10)min'
            elif time_lapsed <= 1200:
                time_lapsed = f'(10 - 20)min'
            elif time_lapsed <= 1800:
                time_lapsed = f'(20 - 30)min'
            elif time_lapsed <= 3600:
                time_lapsed = f'(30 - 59)min'
            else:
                time_lapsed = f'( > hour)s'
            if (year, month) in obs_times_dict:
                if time_lapsed in obs_times_dict[(year, month)]:
                    obs_times_dict[(year, month)][time_lapsed] += 1
                else:
                    obs_times_dict[(year, month)][time_lapsed] = 1
            else:
                obs_times_dict[(year, month)] = {time_lapsed : 1}
        
        obs_times_df = pd.DataFrame.from_dict(obs_times_dict).T
        self.summary_logger.info("Table of monthly time elapsed between observations (NaN included pre resampling): \n" + str(obs_times_df))
        
    def plot_daily_seasonality(self):
        """
        Plotting daily seasonality aggregated by the average of daily values.
        """
        for col in self.data_frame.columns:
            resample_h = self.data_frame[col].resample('1D').mean()
            fig = px.scatter(resample_h, x=resample_h.index, y=f'{col}', title=f'Daily Seasonality of {col}')
            fig.update_layout(xaxis_title='Date', yaxis_title=col)
            fig.show()

    def visualize_seasonality(self):
        for col in self.data_frame.columns:
            resample_h = self.data_frame[[col]].resample('1D').mean()
            resample_h['Month'] = resample_h.index.month
            fig = px.box(resample_h, x='Month', y=col,
                        title=f'Hourly {col} Distribution by Month',
                        labels={'Month': 'Month', 'value': col})
            fig.update_layout(xaxis_title='Month', yaxis_title=col)
            fig.update_xaxes(tickvals=list(range(1, 13)),
                            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            fig.show()


class SummaryPipeline:
    """
    A class to manage the data summarization pipeline.

    Attributes:
        FILE_TYPE (str): Type of data to summarize.
        YEAR (int): Year for summarizing the data.
        MONTH (int): Month for summarizing the data.

    """
    
    def __init__(self, FILE_TYPE, YEAR = None, MONTH = None):
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH

    def run(self):
        """
        Class method used for streamlining data summary
        for use in the PV Gui application.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'data_summary_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting {self.FILE_TYPE} data summary...")

        caller = DataSummary(
            FILE_TYPE = self.FILE_TYPE,
            YEAR = self.YEAR,
            MONTH = self.MONTH
        )
        caller.run_data_summary()
        caller.summarize_data()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")

    def load(self):
        """
        Class method used for loading previous compiled
        data summary.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'data_summary_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting {self.FILE_TYPE} data summary...")

        caller = DataSummary(
            FILE_TYPE = self.FILE_TYPE
        )
        caller.load_data_summary()
        caller.summarize_data()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")


if __name__ == "__main__":
    SummaryPipeline(
        FILE_TYPE = "Irradiance"
    ).load()

