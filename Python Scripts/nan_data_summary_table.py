#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: NaN Data Summary Class, used to view
    characteristics of missing data. Ouputs are
    stored under the logs/ directory

Python Version: Python 3.10.11 
"""

import os
import pandas as pd
import plotly.express as px
import logging
import time
from tqdm import tqdm

from pv_modules import PVModules
from pv_constants import COL_NAMES


class NaNDataSummaryTable(PVModules):
    def __init__(self, FILE_TYPE, propagate_logger = True):
        self.FILE_TYPE = FILE_TYPE
        self.summary_data = pd.DataFrame()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.propagate = propagate_logger

    def run(self):
        month_data = []

        path_list = PVModules(FILE_TYPE=self.FILE_TYPE,
                              propagate_logger=False).get_path_list()
        
        for path in tqdm(path_list, desc="Processing Months"):
            pv_module = PVModules(FILE_TYPE=self.FILE_TYPE,
                                  data_frame=pd.read_csv(path, sep = "\t|,", engine = 'python'),
                                  propagate_logger= False)
            if pv_module.get_data_frame().empty:
                self.logger.error(f"The path: {path} loaded an empty dataframe.")
                continue
            pv_module.reshape_df()
            pv_module.set_night_values()
            if pv_module.get_file_type() == 'Irradiance':
                pv_module.clean_irradiance_values()
            else:
                pv_module.clean_deger_fixed_values()
            pv_module.resample_df()
            data_frame = pv_module.get_data_frame()
            nan_perc, m_perc, col_name, col_perc = self.summarize_nan(data_frame)
            data_frame = pv_module.get_data_frame()
            data_frame.index = data_frame.index.tz_localize(None)
            month_data += [(data_frame.index[0], nan_perc, m_perc) + col_perc]

        month_data = sorted(month_data, key = lambda index : index[1])
        self.summary_data = pd.DataFrame(month_data, columns = ['Month', 'Total NaN %', 'System Outage NaN %'] + col_name).set_index('Month').sort_index()
        self.logger.info("\n" + str(self.summary_data))
    
    def summarize_nan(self, data_frame):
        total_nan = data_frame.isna().sum().sum()
        total_values = data_frame.size
        mt_count = data_frame.isna().all(axis=1).sum()
        t_perc = round(total_nan / total_values * 100,3)
        mt_perc = round(mt_count * len(data_frame.columns) / total_values * 100,3)
        col_name = []
        col_perc = ()
        for col in COL_NAMES[self.FILE_TYPE]:
            n_miss = len(data_frame.index) if col not in data_frame.columns else data_frame[col].isna().sum()
            perc = round(n_miss / total_values * 100,3)
            col_name += [col]
            col_perc += (perc,)

        return t_perc, mt_perc, col_name, col_perc

    def graph_data(self):
        col = [col for col in self.summary_data.columns if col != 'Month']
        fig = px.line(self.summary_data, x = self.summary_data.index, y = col, title = f"{self.FILE_TYPE}: Percentage of NaN by Month")
        fig.update_xaxes(
            rangeslider_visible = True,
            rangeselector = dict(
                buttons = list([
                    dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
                    dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
                    dict(step = "all")])))
        fig.show()
    
    def save_data(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'{super().get_file_type()}_NaN_All.csv')
        self.summary_data.to_csv(file_path)

    def reload(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'{super().get_file_type()}_NaN_All.csv')
        try:
            self.summary_data = pd.read_csv(file_path, index_col = 'Month').sort_index()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def to_latex(self):
        self.summary_data.index = pd.to_datetime(self.summary_data.index)
        self.summary_data.index = self.summary_data.index.strftime('%B %Y')
        latex_code = self.summary_data.to_latex(index=True)
        print(latex_code)

    def get_summary_data(self):
        return self.summary_data

class NaNSummaryTablePipeline:
    def __init__(self, FILE_TYPE):
        self.FILE_TYPE = FILE_TYPE

    def run(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'nan_summary_table_log.log')
        logging.basicConfig(filename = file_path,
                            level = logging.INFO,
                            format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"Starting summary of {self.FILE_TYPE} for total available data...")

        caller = NaNDataSummaryTable(
            FILE_TYPE = self.FILE_TYPE
        )
        caller.run()
        caller.save_data()
        caller.graph_data()
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Complete.")
        logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
        logger.info("\n\n# ====================================================================== #\n"
                    + "# =============================== New Run ============================== #\n"
                    + "# ====================================================================== #\n")
        
    def load(self):
        caller = NaNDataSummaryTable(
            FILE_TYPE = self.FILE_TYPE,
            propagate_logger=False
        )
        caller.reload()
        caller.graph_data()
        return caller.get_summary_data()


if __name__ == "__main__":
    NaNSummaryTablePipeline(
        FILE_TYPE = "Irradiance"
    ).run()