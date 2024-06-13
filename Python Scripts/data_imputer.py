#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: Data Imputer Class, used to impute missing values
    with methods developed for this project. Ouputs individual
    monthly clean data files.

Python Version: Python 3.10.11 
"""

import os
import pandas as pd
import numpy as np
import re
import threading
import queue
import logging
import time

from pv_modules import PVModules
from interpolate import Interpolate
from arima import ARIMA


class Imputer(PVModules):
    """
    Class for data imputation and cleaning.

    Attributes:
        Parent Attributes:
            - month (str): Month.
            - year (str): Year.
            - file (str): File type.
            - data_frame (pandas DataFrame): 
        Child attributes:
            - input_data_frame (pandas DataFrame): Input DataFrame.

    Methods:
        - run: Runs the data imputation and cleaning process.
    """
    def __init__(self, FILE_TYPE, YEAR, MONTH, data_frame, propagate_log = True):
        super().__init__(FILE_TYPE = FILE_TYPE, data_frame = data_frame)
        self.clean_data_frame = pd.DataFrame()
        self.YEAR = YEAR
        self.MONTH = MONTH

        self.nan_gaps = {}
        self.imputer_log = logging.getLogger(type(self).__name__)
        self.imputer_log.propagate = propagate_log

    def run(self):
        """
        Wrapper function for loading and preprocessing the data frame
        for the imputation of missing values.

        """
        # === reshaping df for timestap & adjusted headers === #
        super().reshape_df()
        # === Removing Errors === #
        super().clean_irradiance_values()
        # === Using PvLib to remove nightime values === #
        super().set_night_values()
        # === resample df to 20s frequency === #
        super().resample_df()
        self.clean_data_frame = super().get_data_frame()
        # === Impute missing values === #
        self.imputer_log.info(f"Imputing {round(self.clean_data_frame.isna().sum().sum() / self.clean_data_frame.size * 100, 3)}% of the data for {self.MONTH}, {self.YEAR}.")
        self.impute_missing_values()
        if self.clean_data_frame.isna().sum().sum():
            self.imputer_log.error(f"Error - The File {self.FILE_TYPE}, {self.MONTH} {self.YEAR} still has NaN values")
        else:
            # self.save_cleaned_data_frame()
            pass


    def impute_missing_values(self):
        """
        Finds the index positions of gaps in a DataFrame based on
        their size and applies appropriate imputation method.

        Args:
            - clean_data_frame (pandas.DataFrame): Input DataFrame.

        Returns:
            - tuple: Dictionaries containing index positions of
              NaN values classified based on their gap size.
        """
        self.interpolation = {}
        self.arima = {}
        self.rnn = {}
        knn = {}
        
        for col in self.clean_data_frame.columns:
            nan_indices = np.where(pd.isna(self.clean_data_frame[col]))[0]
            if len(nan_indices) == 0:
                continue
            
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

            timestamps_consecutive_segments = [[self.clean_data_frame.index[index] for index in segment] for segment in consecutive_segments]
            
            for segment in timestamps_consecutive_segments:
                start_index = segment[0]
                end_index = segment[-1]
                delta_time = (end_index - start_index).total_seconds()
                if start_index == self.clean_data_frame.index[0]:
                    if not col in self.rnn: self.rnn[col] = []
                    self.rnn[col] += [segment]
                elif delta_time < 100:
                    if not col in self.interpolation: self.interpolation[col] = []
                    self.interpolation[col] += [segment]
                elif delta_time < 1000:
                    if not col in self.arima: self.arima[col] = []
                    self.arima[col] += [segment]
                elif delta_time < 10000:
                    if not col in self.rnn: self.rnn[col] = []
                    self.rnn[col] += [segment]
                else:
                    self.imputer_log.error("Gap is larger than 10000")

        self.interpolation_method()
        self.arima_method() 
                    

    def interpolation_method(self):
        """
        Performs interpolation on a DataFrame to fill missing
        values using the 'time' method.

        Args:
            - df (pandas.DataFrame): Input DataFrame.
            - nan_gaps (dict): Dictionary containing column names
            - as keys and lists of NaN gap indices as values.

        Returns:
            - pandas DataFrame: DataFrame with filled values using
              time based linear interpolation.
            
        """
        for col in self.interpolation:
            imputer = Interpolate(self.clean_data_frame[col], propagate_log = False)
            imputer.impute()
            flattened_list = list(self.flatten_list(self.interpolation[col]))
            self.clean_data_frame[col].loc[flattened_list] = imputer.get_imputed_values().loc[flattened_list]

    def arima_method(self):
        """
        Imputes missing values using the ARIMA Class developed
        for this project.

        Args:
            - df (pandas.DataFrame): Input DataFrame.
            - nan_gaps (dict): Dictionary containing column names
            - as keys and lists of NaN gap indices as values.

        Returns:
            - pandas DataFrame: DataFrame with filled values using
              autoregressive integrated moving average method of imputation.
        """
        for col in self.arima:
            for gap in self.arima[col]:
                num_nan_values = len(gap)
                first_nan_index = gap[0]
                X = np.array(self.clean_data_frame[col].loc[:first_nan_index])[:-1]
                p = 2 * num_nan_values
                q = 1 * num_nan_values
                imputer = ARIMA(p=p,q=q)
                imputer.fit(X)
                imputed_values = imputer.forecast(steps = num_nan_values)
                self.clean_data_frame[col].loc[gap] = imputed_values

    def flatten_list(self, nested_list):
        """
        Helper function for the impute_missing_values method. Used
        to flatten the nested lists that's outputed from finding the
        different NaN gaps.

        Args:
            nested_list (list): Nested staggered list of NaN gaps.

        Returns:
            (list): flattened list 
        
        """
        for item in nested_list:
            if isinstance(item, list):
                yield from self.flatten_list(item)
            else:
                yield item

    def save_cleaned_data_frame(self):
        file_name = 'clean_' + self.MONTH.lower() + '.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', self.YEAR, self.FILE_TYPE, file_name)
        self.clean_data_frame.to_csv(file_path)


class Threader(threading.Thread):
    """
    Thread worker class for parallel processing.

    Attributes:
        - queue (Queue): Queue containing file paths.
        - file (str): File type.
        - lock (threading Lock): Lock for thread synchronization.
        - threading_logger (logging.logger): logger object
        - propagate (bool): flag indicating whether to propagate messages to logger
    """
    def __init__(self, QUEUE, LOCK, FILE_TYPE, propagate_logger = True):
        """
        Threader constructor to initalize threads.
            
        """
        threading.Thread.__init__(self)
        self.queue = QUEUE
        self.lock = LOCK
        self.FILE_TYPE = FILE_TYPE
        self.threading_logger = logging.getLogger(type(self).__name__)
        self.threading_logger.propagate = propagate_logger

    def run(self):
        """
        
        """
        while True:
            try:
                file_path = self.queue.get(timeout = 3) # retrieve file path from the queue
            except queue.Empty:
                return # If the queue is empty, exit the thread

            year, month = re.search(r"/(\d{4})/[a-zA-Z]*/([a-zA-Z]*)\.csv",file_path).group(1,2)
            df = pd.read_csv(file_path, sep="\t|,", engine='python')
            self.lock.acquire()
            self.threading_logger.info(f"Starting {month}, {year}")
            self.lock.release()
            Imputer(FILE_TYPE = self.FILE_TYPE,
                    YEAR = year,
                    MONTH = month, 
                    data_frame = df,
                    propagate_log = True).run()
            self.lock.acquire()
            self.threading_logger.info(f"Completed {month}, {year}")
            self.lock.release()
            self.queue.task_done()


class ImputerPipeline:
    """
    Class pipeline for running the imputer program.
    
    Attributes:
        - FILE_TYPE (str): File type (Irradiance/Deger/Fixed)
        - YEAR (str): Year of file
        - MONTH (str): Month of file
    
    Methods:
        - run:
        
    """
    def __init__(self, FILE_TYPE, YEAR = None, MONTH = None):
        """
        ImputerPipeline constructor, used to intialize intance of class.
        
        """
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH

    def run(self):
        """
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'data_imputation_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting Data Imputation Program on {self.FILE_TYPE}{f''' for {self.MONTH}, {self.YEAR}''' if (self.FILE_TYPE and self.MONTH) else ''''''}...")

        q = queue.Queue()
        file_paths = PVModules(
                        FILE_TYPE = self.FILE_TYPE,
                        YEAR = self.YEAR,
                        MONTH = self.MONTH
                    ).path_list
        for file_path in file_paths:
            q.put_nowait(file_path)

        lock = threading.Lock()
        num_of_processors = 6 if len(file_path) >= 6 else len(file_path)
        for _ in range(num_of_processors): 
            t = Threader(
                QUEUE=q,
                FILE_TYPE=self.FILE_TYPE,
                LOCK=lock
                )
            t.daemon = True
            t.start()

        q.join() 

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")

        
if __name__ == "__main__":
    ImputerPipeline(
        FILE_TYPE="Irradiance"
    ).run()