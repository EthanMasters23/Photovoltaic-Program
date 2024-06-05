#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: Imputation Methid Testing Class, basically
    serves as a pipeline for testing different imputation
    methods with different parameters to decide functionality
    of the data imputer program.

Python Version: Python 3.10.11 
"""

import os
import pandas as pd
import logging
import time
from imputation_method_modules import ImputationModules
from pv_modules import PVModules
from tqdm import tqdm


class ImputationMethodTesting(ImputationModules):
    def __init__(self, GAP_LENGTH, ITERATIONS, FILE_TYPE, YEAR = None, MONTH = None, propgagate_logger = True):
        super().__init__(FILE_TYPE)
        self.GAP_LENGTH = GAP_LENGTH
        self.ITERATIONS = ITERATIONS
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH
        self.performance_data_frame = pd.DataFrame()
        self.testing_logger = logging.getLogger(type(self).__name__)
        self.testing_logger.propagate = propgagate_logger

    def run_performance_test(self):
        self.testing_logger.info(f"Starting imputation method testing for gap-size {self.GAP_LENGTH} (seconds).")
        pv_modules = PVModules(FILE_TYPE = self.FILE_TYPE, YEAR = self.YEAR, MONTH = self.MONTH, propagate_logger=False)
        path_list = pv_modules.get_path_list()
        path = path_list[0]
        for _ in tqdm(range(self.ITERATIONS)):
            super().set_path(path)
            super().clean_data_frame()
            super().simulate_missing_values(self.GAP_LENGTH)
            super().missing_data_edge_case()
            super().map_nan_gap_indices()
            self.run_imputation_methods()
            if self.performance_data_frame.empty:
                self.performance_data_frame = self.imputation_methods_data_frame.copy()
            else:
                self.performance_data_frame = (self.performance_data_frame + self.imputation_methods_data_frame.copy()) / 2

        self.run_performance_metrics()


    def run_imputation_methods(self):
        super().interpolation()
        super().interpolation_method()
        # super().knn_method()
        super().arima_method()
        super().rnn_method()
        self.imputation_methods_data_frame = super().get_imputation_methods_data()

    def run_performance_metrics(self):
        r2_df = pd.DataFrame(self.performance_data_frame.drop(['mae','mse','rmse'], axis = 0, level = 1).max(axis = 0),columns = ['r2'])
        r2_df['Method'] = self.performance_data_frame.drop(['mae','mse','rmse'], axis = 0, level = 1).idxmax(axis = 0).values
        mae_df = pd.DataFrame(self.performance_data_frame.drop(['r2','mse','rmse'], axis = 0, level = 1).min(axis=0),columns = ['mae'])
        mae_df['Method'] = self.performance_data_frame.drop(['r2','mse','rmse'], axis = 0, level = 1).idxmin(axis=0).values
        rmse_df = pd.DataFrame(self.performance_data_frame.drop(['r2','mse','mae'], axis = 0,level = 1).min(axis=0),columns = ['rmse'])
        rmse_df['Method'] = self.performance_data_frame.drop(['r2','mse','mae'], axis = 0, level = 1).idxmin(axis=0).values
        mse_df = pd.DataFrame(self.performance_data_frame.drop(['r2','rmse','mae'], axis = 0, level = 1).min(axis=0),columns = ['mse'])
        mse_df['Method'] = self.performance_data_frame.drop(['r2','rmse','mae'], axis = 0, level = 1).idxmin(axis=0).values
        for row in r2_df.index:
            self.testing_logger.info(row)
            self.testing_logger.info(f"Optimal Imputation Method for {row}: {r2_df.loc[row]['Method'][0]}, R2 score: {round(r2_df.loc[row]['r2'],5)}")
            self.testing_logger.info(f"Optimal Imputation Method for {row}: {mae_df.loc[row]['Method'][0]}, MAE score: {round(mae_df.loc[row]['mae'],5)}")
            self.testing_logger.info(f"Optimal Imputation Method for {row}: {rmse_df.loc[row]['Method'][0]}, RMSE score: {round(rmse_df.loc[row]['rmse'],5)}")
            self.testing_logger.info(f"Optimal Imputation Method for {row}: {mse_df.loc[row]['Method'][0]}, MSE score: {round(mse_df.loc[row]['mse'],5)}")
        
        self.testing_logger.info("Performance DataFrame:")
        self.testing_logger.info("\n" + str(self.performance_data_frame))


class MethodTestingPipeline:
    def __init__(self, GAP_LENGTH, ITERATIONS, FILE_TYPE, YEAR, MONTH):
        self.GAP_LENGTH = GAP_LENGTH
        self.ITERATIONS = ITERATIONS
        self.FILE_TYPE = FILE_TYPE
        self.YEAR = YEAR
        self.MONTH = MONTH

    def run(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'imputation_method_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting main script...")

        # ====== Main Caller ====== #
        pipeline = ImputationMethodTesting(
            GAP_LENGTH = self.GAP_LENGTH,
            ITERATIONS = self.ITERATIONS,
            FILE_TYPE = self.FILE_TYPE,
            YEAR = self.YEAR,
            MONTH = self.MONTH
        )
        # ========================= #

        pipeline.run_performance_test()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #")


if __name__ == "__main__":
    pipeline = MethodTestingPipeline(
        GAP_LENGTH = 80, # (seconds)
        ITERATIONS = 10, # How many imputation runs with random gaps each iteration
        FILE_TYPE = "Irradiance", # (opt: Irradiance/Deger/Fixed): 
        YEAR = "2022", # (format: YYYY)
        MONTH = "jul" # (format: jul)
    ).run()