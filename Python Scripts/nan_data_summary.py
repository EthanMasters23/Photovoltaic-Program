#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: NaN Data Summary Class, used for further in-depth
    analysis of missing data characteristics. Ouputs are stored
    in the logs/ directory

Python Version: Python 3.10.11 
"""

import os
import pandas as pd
import numpy as np
import logging
import time

from data_compiler import DataCompiler
from pv_modules import PVModules


class NaNSummary:
    """
    A class for summarizing NaN values and NaN gaps in data.

    Attributes:
        - FILE_TYPE (str): The type of file being analyzed.
        - YEAR (int): The year of the data.
        - MONTH (int): The month of the data.
        - propagate_logger (bool): Flag to propagate logger messages.
    
    Methods:
        - run: Starts the NaN Summary Program by loading all available data for the
        - given FILE_TYPE attribute.
        - run_momth: Starts the NaN Summary Program for a given MONTH and FILE_TYPE attribute.
    """

    def __init__(self, FILE_TYPE = None, YEAR = None, MONTH = None, propagate_logger = True):
        """
        Class constructor. Initializes an instance of NaNSummary.

        Args:
            - FILE_TYPE (str, optional): The type of file being analyzed. Defaults to None.
            - YEAR (int, optional): The year of the data. Defaults to None.
            - MONTH (int, optional): The month of the data. Defaults to None.
            - propagate_logger (bool, optional): Flag to propagate logging messages. Defaults to True.
        """
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH

        self.summary_logger = logging.getLogger(type(self).__name__)
        self.summary_logger.propagate = propagate_logger
        self.summary_data = pd.DataFrame()

    def run(self):
        """
        Run the NaN summary for all available data.
        """
        self.summary_logger.info(f"Summary for all available {self.FILE_TYPE} data.")
        data_compiler_clean = DataCompiler(FILE_TYPE = self.FILE_TYPE, FILE_OPTION = 'processed')
        data_compiler_clean.load_compiled_data()
        self.summary_data = data_compiler_clean.get_compiled_data()
        self.summary_data = PVModules.add_time_features(self.summary_data)

    def run_month(self):
        """
        Run the NaN summary for a specific month of data.
        """
        self.summary_logger.info(f"Summary for {self.FILE_TYPE} {self.YEAR}-{self.MONTH}.")
        pv_module = PVModules(FILE_TYPE=self.FILE_TYPE, MONTH=self.MONTH, YEAR=self.YEAR)
        path = pv_module.get_path_list()
        data_frame = pd.read_csv(path, sep = "\t|,", engine = 'python')
        pv_module.set_data_frame(data_frame)
        pv_module.reshape_df()
        if pv_module.get_file_type() == 'Irradiance':
            pv_module.clean_irradiance_values()
        else:
            pv_module.clean_deger_fixed_values()
        pv_module.set_night_values()
        pv_module.resample_df()
        pv_module.time_features()
        self.summary_data = pv_module.get_data_frame()


    def summarize_nan(self):
        """
        Summarize NaN values in the data.
        """
        self.summary_logger.info("------------------------ Summary of NaN Values ------------------------")
        total_nan = self.summary_data.drop(['day','month','year'],axis=1).isna().sum().sum()
        total_values = self.summary_data.drop(['day','month','year'],axis=1).size
        mt_count = self.summary_data.drop(['day','month','year'],axis=1).isna().all(axis=1).sum()
        t_perc = round(total_nan / total_values * 100,3)
        mt_perc = round(mt_count * (len(self.summary_data.columns)-3) / total_values * 100, 3)
        self.summary_logger.info(f"Percentage of NaN values due to System Outage: {mt_perc}%")
        self.summary_logger.info(f"Precentage of MAR NaN values: {round(t_perc-mt_perc, 3)}%")
        self.summary_logger.info(f"Precentage of Total NaN values: {t_perc}%")

        self.summary_logger.info("--- Missing values by variable ---")
        for col in self.summary_data.columns:
            if not col in ['day', 'month', 'year']:
                n_miss = self.summary_data[col].isna().sum()
                perc = round(n_miss / self.summary_data.shape[0] * 100, 3)
                self.summary_logger.info(f"{col}, Missing: {n_miss} ({perc}%)")

        self.summary_logger.info("--- Missing values by day ---")

        for row in self.summary_data['day'].unique():
            n_miss = self.summary_data[self.summary_data['day']==row].drop(['day', 'month', 'year'], axis = 1).isna().sum().sum()
            perc = round(n_miss / self.summary_data[self.summary_data['day']==row].drop(['day', 'month', 'year'], axis = 1).size * 100,3)
            self.summary_logger.info(f"{row}, Missing: {n_miss} ({perc}%)")
        
        if len(self.summary_data['month'].unique()) == 1: return
        
        self.summary_logger.info("--- Missing values by month ---")    

        for row in sorted(self.summary_data['month'].unique()):
            n_miss = self.summary_data[self.summary_data['month']==row].drop(['day','month','year'],axis=1).isna().sum().sum()
            perc = round(n_miss / self.summary_data[self.summary_data['month']==row].drop(['day','month','year'],axis=1).size * 100,3)
            self.summary_logger.info(f"{row}, Missing: {n_miss} ({perc}%)")

        self.summary_logger.info("--- Missing values by year ---")    

        for row in self.summary_data['year'].unique():
            n_miss = self.summary_data[self.summary_data['year']==row].drop(['day', 'month', 'year'], axis = 1).isna().sum().sum()
            perc = round(n_miss / self.summary_data[self.summary_data['year']==row].drop(['day', 'month', 'year'], axis = 1).size * 100,3)
            self.summary_logger.info(f"{row}, Missing: {n_miss} ({perc}%)")


    def summarize_nan_gaps(self):
        """
        Summarize NaN gaps in the data.
        """
        self.summary_logger.info("------------------------ Summary of NaN Gaps ------------------------")
        nan_gaps = []
        for col in range(len(self.summary_data.drop(['day','month','year'],axis=1).columns)):
            column_data = self.summary_data.iloc[:, col]
            nan_indices = column_data.index[column_data.isnull()]
            start_time = nan_indices[0]
            for i, index in enumerate(nan_indices):
                if i != len(nan_indices) - 1 and (nan_indices[i+1] - index).total_seconds() != 20:
                    nan_gaps += [[
                        (index - start_time).total_seconds(),
                        start_time,
                        index,
                        self.summary_data.columns[col]
                    ]]
                    start_time = nan_indices[i+1]

        nan_df = pd.DataFrame(nan_gaps, columns = ['Seconds', 'Start Time', 'End Time', 'Column'])
        grouped = nan_df.groupby('Column')

        for column_value, group_data in grouped:
            arr = group_data['Seconds']
            z_scores = np.abs((arr - arr.mean()) / arr.std())
            threshold = 2
            outliers = arr[z_scores > threshold]
            num_of_gaps = len(group_data['Seconds'])
            value_counts = group_data['Seconds'].value_counts()
            num_gaps_range = len(group_data[(group_data['Seconds'] > 60) & (group_data['Seconds'] <= 120)])
            outlier_num_gaps = len(group_data[group_data['Seconds'] > 120])
            self.summary_logger.info(f"--- Statistics for {column_value} ---")
            self.summary_logger.info(f"Number of gaps: {num_of_gaps}")
            self.summary_logger.info(f"Percentage of single observation missing: {round((value_counts.get(0, 0) / num_of_gaps) * 100, 3)}%")
            self.summary_logger.info(f"Percentage of 20s gaps: {round((value_counts.get(20, 0) / num_of_gaps) * 100, 3)}%")
            self.summary_logger.info(f"Percentage of 40s gaps: {round((value_counts.get(40, 0) / num_of_gaps) * 100, 3)}%")
            self.summary_logger.info(f"Percentage of 60s gaps: {round((value_counts.get(60, 0) / num_of_gaps) * 100, 3)}%")
            self.summary_logger.info(f"Percentage of (60s, 120s] gaps: {round((num_gaps_range / num_of_gaps) * 100, 3)}%")
            self.summary_logger.info(f"Percentage of gaps larger than 120s: {round((outlier_num_gaps / num_of_gaps) * 100, 3)}%")
            if len(outliers):
                self.summary_logger.info(f"Number of outliers: {len(outliers)}, min (seconds): {sorted(outliers)[0]}, max (seconds): {sorted(outliers)[-1]}")

class NaNSummaryPipeline:
    """
    A Class pipeline for running the NaN summary program.

    Attributes:
        - FILE_TYPE (str): file type (Irradiance/Deger/Fixed)
        - YEAR (str): Year of File
        - MONTH (str): Month of File

    Methods:
        - run: Runs the NaN Summary Program.
    """
    def __init__(self, FILE_TYPE, YEAR = None, MONTH = None):
        """
        Class constructor. Initializes an instance of NaNSummaryPipeline.

        Args:
            - file_type (str): The type of file being analyzed.
            - year (int, optional): The year of the data. Defaults to None.
            - month (int, optional): The month of the data. Defaults to None.
        """
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH

    def run(self):
        """
        Run the NaN summary program.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'nan_data_summary_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting NaN Data Summary Program on {self.FILE_TYPE}{f''' for {self.MONTH}, {self.YEAR}''' if (self.YEAR and self.MONTH) else ''''''}...")

        caller = NaNSummary(
            FILE_TYPE = self.FILE_TYPE, 
            YEAR = self.YEAR, 
            MONTH = self.MONTH 
        )

        if not (self.YEAR and self.MONTH):
            caller.run()
        else:
            caller.run_month()
        caller.summarize_nan()
        caller.summarize_nan_gaps()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")




if __name__ == "__main__":
    caller = NaNSummaryPipeline(
        file_type= 'Irradiance'
    )
    caller.run()