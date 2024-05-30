#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on April 20 18:50 2023

@author: Ethan Masters

Purpose: Data Compiler Class, used to generate a single
    file that stores all available data for a given file
    class (Irradiance, Deger, Fixed)

Python Version: Python 3.10.11 
"""

import os
import pandas as pd
import time
import logging
from tqdm import tqdm
from pv_modules import PVModules


class DataCompiler:
    """
    A class to compile various data types into one CSV file.

    Attributes:
        FILE_TYPE (str): Type of data to compile.
        FILE_OPTION (str): Option for processing the compiled data.
        compiled_data (pandas.DataFrame): DataFrame containing compiled data.
        file (str): Name of the compiled CSV file.
        compiler_logger (logging.Logger): Logger for data compilation operations.
        prop_log (bool): Flag to propagate logging messages.

    """

    def __init__(self, FILE_OPTION, FILE_TYPE, propagate_log = True):
        """
        Initializes DataCompiler class with provided parameters.

        Args:
            FILE_TYPE (str): Type of data to compile.
            FILE_OPTION (str): Option for processing the compiled data.
            propagate_log (bool): Flag to propagate logging messages.

        """
        self.FILE_TYPE = FILE_TYPE
        self.FILE_OPTION = FILE_OPTION
        self.compiled_data = pd.DataFrame()
        self.file = f'{self.FILE_TYPE}_compiled_{self.FILE_OPTION}.csv'
        self.compiler_logger = logging.getLogger(type(self).__name__)
        self.compiler_logger.propagate = propagate_log
        self.prop_log = propagate_log

    def run(self):
        """
        Executes the data compilation process.

        This method iterates through all available data files for a given data type, loads each file,
        preprocesses the data based on the FILE_OPTION, and compiles the preprocessed data into a single DataFrame.

        Raises:
            Exception: If any loaded data file results in an empty DataFrame.

        """
        pv_modules = PVModules(FILE_TYPE = self.FILE_TYPE, propagate_logger = self.prop_log)
        for path in tqdm(pv_modules.get_path_list()):
            pv_modules.set_data_frame(data_frame = pd.read_csv(path, sep = "\t|,", engine = 'python'))
            if pv_modules.get_data_frame().empty:
                raise Exception(f"The path: {path} loaded an empty dataframe.")
            pv_modules.reshape_df()
            if self.FILE_OPTION == 'processed':
                if pv_modules.get_file_type() == 'Irradiance':
                    pv_modules.clean_irradiance_values()
                else:
                    pv_modules.clean_deger_fixed_values()
                pv_modules.set_night_values()
                pv_modules.resample_df()
            self.compiled_data = pd.concat([self.compiled_data,
                                           pv_modules.get_data_frame().copy()],
                                           axis = 0,
                                           ignore_index = False).sort_index()

    def save_compiled_data(self):
        """
        Saves the compiled data to a CSV file.

        This method saves the compiled DataFrame to a CSV file in the 'assets' directory.

        """
        self.compiler_logger.info(f"Saving data frame as {self.file}")
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', self.file)
        self.compiled_data.to_csv(file_path)

    def load_compiled_data(self):
        """
        Loads the compiled data from a CSV file.

        This method loads the compiled data from a previously saved CSV file in the 'assets' directory.

        Raises:
            FileNotFoundError: If the specified file is not found.

        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', self.file)
        try:
            self.compiled_data = pd.read_csv(file_path, index_col = 0)
            self.compiled_data.index = pd.to_datetime(self.compiled_data.index)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
    def get_compiled_data(self):
        """
        Gets the compiled data.

        Returns:
            pandas.DataFrame: Compiled DataFrame.

        """
        return self.compiled_data


class CompilerPipeline:
    """
    A class to manage the data compilation pipeline.

    Attributes:
        FILE_TYPE (str): Type of data to compile.
        FILE_OPTION (str): Option for processing the compiled data.

    """
    def __init__(self, FILE_OPTION, FILE_TYPE):
        """
        Initializes CompilerPipeline class with provided parameters.

        Args:
            FILE_TYPE (str): Type of data to compile.
            FILE_OPTION (str): Option for processing the compiled data.

        """
        self.FILE_OPTION = FILE_OPTION
        self.FILE_TYPE = FILE_TYPE

    def run(self):
        """
        Executes the data compilation pipeline.

        This method orchestrates the entire data compilation process by creating a DataCompiler instance,
        executing the data compilation, and logging the process.

        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'data_compiler_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting data compiler for {self.FILE_TYPE}...")

        caller = DataCompiler(
            FILE_OPTION = self.FILE_OPTION,
            FILE_TYPE = self.FILE_TYPE
        )
        caller.run()
        caller.save_compiled_data()
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")
    
    def load(self):
        """
        Loads the compiled data.

        This method loads the compiled data from a CSV file and returns it as a DataFrame.

        Returns:
            pandas.DataFrame: Compiled DataFrame containing the preprocessed data.

        """
        caller = DataCompiler(
            self.FILE_OPTION,
            self.FILE_TYPE,
            propagate_log=False
        )
        caller.load_compiled_data()
        return caller.get_compiled_data()


if __name__ == "__main__":
    CompilerPipeline(
        FILE_OPTION = 'processed', # (opt: processed/raw)
        FILE_TYPE = 'Irradiance' # (opt: Irradiance/Deger/Fixed)
    ).run()