#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on June 12 16:03 2023

@author: Ethan Masters

Purpose: KNN Custom Class

Python Version: Python 3.10.11 (main, Apr 20 2023, 13:59:00)
"""

import logging
import numpy as np
import pandas as pd
from feature_engineering import FeatureEngineering
from pv_constants import DATA_SUMMARY_DICT


class KNN:
    """
    A class to perform k-nearest neighbors (KNN) imputation.

    Attributes:
        k_neighbors (int): Number of neighbors to consider.
        training_data (pandas.DataFrame): DataFrame containing training data.
        feature_data (pandas.DataFrame): DataFrame containing feature data.
        knn_logger (logging.Logger): Logger for KNN operations.

    """

    def __init__(self, k, propagate_log = True):
        """
        Initializes KNN class with provided parameters.

        Args:
            k (int): Number of neighbors to consider.
            propagate_log (bool): Flag to propagate logging messages.

        """
        self.k_neighbors = k
        self.load_training_data()
        self.knn_logger = logging.getLogger(type(self).__name__)
        self.knn_logger.propagate = propagate_log

    def load_training_data(self):
        """
        Load training data using FeatureEngineering Class.

        """
        feature_engineerer = FeatureEngineering(propagate_log=False)
        feature_engineerer.load_features_data_frame()
        feature_engineerer.load_training_data_frame()
        self.training_data = feature_engineerer.get_training_data()[self.col_nam]
        self.feature_data = feature_engineerer.get_features_data_frame()

    def impute(self, target_y_index, col_name):
        """
        Perform KNN imputation.

        Args:
            target_y_index (numpy.ndarray): Indices of the target values to impute.
            col_name (str): Name of the column to impute.

        """
        self.knn_logger.info(f"Starting KNN Method ({self.col_nam}): Using {self.k_neighbors} neighbors and removing night values.")
        self.target_y_index = target_y_index
        self.col_nam = col_name
        self.training_Y = None
        self.training_X = None
        self.target_X = None
        self.target_Y = np.empty(len(target_y_index))
        self.build_training_and_target_data()
        self.scale_data()
        self.fit_transform()
        self.knn_logger.info("Finished KNN Imputer.")
    
    def build_training_and_target_data(self):
        """
        Build training and target data sets.

        """
        self.training_Y = self.training_data.drop(self.target_y_index).to_numpy()
        self.training_X = self.feature_data.drop(self.target_y_index).to_numpy()
        self.target_X = self.feature_data.loc[self.target_y_index].to_numpy()

    def scale_data(self):
        """
        Scale the data between 1 and 100.

        Args:
            data (numpy.ndarray): Data to scale.

        Returns:
            numpy.ndarray: Scaled data.

        """
        self.training_X = self.scaler(self.training_X)
        self.target_X = self.scaler(self.target_X)

    def pairwise_euclidean_distance(self):
        """
        Calculate the pairwise Euclidean distances between points in the array.

        """
        broadcasted_arr = np.tile(self.target_row, (self.training_X.shape[0], 1))
        self.distances = np.sqrt(np.sum(np.square(broadcasted_arr - self.training_X), axis = 1))

    def sort_neighbors(self):
        """
        Sort distances and find the k number of neighbors.

        """
        indexed_arr = [(value, index) for index, value in enumerate(self.distances)]
        sorted_arr = sorted(indexed_arr, key = lambda x: x[0])
        index_list = [index[1] for index in sorted_arr[:self.k_neighbors]]
        self.k_neighbors = self.training_Y[index_list]
    
    def average_distances(self):
        """
        Averages self.k_neighbors (np.array) to find self.target_value.

        """
        self.target_value = np.mean(self.k_neighbors)

    def fit_transform(self):
        """
        Fit the model and transform the target data.

        """
        for index, value in enumerate(self.target_X):
            self.target_row = value
            self.pairwise_euclidean_distance()
            self.sort_neighbors()
            self.average_distances()
            self.target_Y[index] = round(self.target_value,3)

    def get_imputed_values(self):
        """
        Geter method for imputed target values.

        Returns:
            numpy.ndarray: Imputed target values.

        """
        return self.target_Y
    
    def scaler(self, data):
        """
        Scale the data between 1 and 100 using min-max scaling.

        Args:
            data (numpy.ndarray): Data to scale.

        Returns:
            numpy.ndarray: Scaled data.

        """
        min_val = DATA_SUMMARY_DICT[self.col_nam]['min']
        max_val = DATA_SUMMARY_DICT[self.col_nam]['max']
        scaled_data = 1 + (data - min_val) * (99 / (max_val - min_val))
        return scaled_data