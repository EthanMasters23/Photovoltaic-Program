"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: Custom RNN Class

Python Version: Python 3.10.11 
"""

import json
import numpy as np
import os
import logging
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pv_constants import DATA_SUMMARY_DICT
from pv_modules import PVModules
import time
import pandas as pd
import pprint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Concatenate, Input, TimeDistributed
from tensorflow.keras.models import Model

# == set for reproducability == #
np.random.seed(42)


class RNN:
    """
    Recurrent Neural Network

    Attributes:
        input_data (pandas DataFrame)

    Methods:


    """
    def __init__(self, input_data = None, variable_name = None):
        self.input_data = input_data
        self.variable_name = variable_name
        self.rnn_logger = logging.getLogger(type(self).__name__)

    def compile(self):
        """
        Wrapper function

        Used to compile and build the RNN Model.
        """
        self.preprocess_data()
        self.build_model()

    def preprocess_data_bidirectional(self):
        """
        Method used to process training data for bidirectional RNN model.
        
        """
        self.X_forward, self.X_backward, self.y = self.create_sequences()
        self.X_train_forward, self.X_val_forward, self.X_train_backward, self.X_val_backward, self.y_train, self.y_val = train_test_split(
            self.X_forward, self.X_backward, self.y, test_size=0.2, random_state=42, shuffle=True)
        self.X_train_forward, self.X_val_forward, self.X_train_backward, self.X_val_backward, self.y_train, self.y_val = (
            self.scale_data(self.X_train_forward),
            self.scale_data(self.X_val_forward),
            self.scale_data(self.X_train_backward),
            self.scale_data(self.X_val_backward),
            self.scale_data(self.y_train),
            self.scale_data(self.y_val)
        )

    def preprocess_data(self):
        """
        Method used to process training data for forward directional RNN Model.
        
        """
        self.X, self.y = self.create_sequences()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=True)
        self.X_train, self.X_val, self.y_train, self.y_val = (
            self.scale_data(self.X_train),
            self.scale_data(self.X_val),
            self.scale_data(self.y_train),
            self.scale_data(self.y_val)
        )

    def create_sequences_bidrectional(self):
        """
        Generates sequences for the bidirectional model.
        
        """
        input_data = np.array(self.input_data)
        num_samples = input_data[0].shape[0] // 2
        steps_removed = 2 + (input_data[0].shape[0] % 2)
        self.time_steps = num_samples - steps_removed
        self.num_features = 1
        
        X_forward = input_data[:, : - (num_samples + steps_removed)] 
        X_backward = input_data[:, num_samples + steps_removed:] 

        print(input_data[0], '\n', X_forward[0], '\n', X_backward[0])

        X_forward = X_forward.reshape(X_forward.shape[0], X_forward.shape[1], 1)
        X_backward = X_backward.reshape(X_backward.shape[0], X_backward.shape[1], 1)
        
        y = input_data[:, num_samples - steps_removed + 1: - (num_samples - steps_removed + 1)]
        # y = y.reshape(y.shape[0], y.shape[1], 1)

        self.predicted_steps = y[0].shape[0]
        print("Y", y[0].shape)
        self.input_shape = (self.time_steps + 1, 1)
        print(input_data[0], '\n', X_forward[0], '\n', X_backward[0], '\n', y[0])
        del self.input_data, input_data
        return X_forward, X_backward, y
    
    def create_sequences(self):
        """
        Generate sequenes for the forward directional model.
        
        """
        input_data = np.array(self.input_data)
        print("input: ", input_data.shape)
        self.predicted_steps = 5
        self.num_features = 1
        X = input_data[:, : - self.predicted_steps] 
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = input_data[:, -self.predicted_steps :]
        self.input_shape = (X[0].shape[0], 1)
        del self.input_data, input_data
        return X, y

    def build_model(self):
        """
        Builds and compiles the model for training.

        """
        model = ModelConfig().build_model_v5(self.input_shape, self.predicted_steps)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07)
        file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'rnn_model', datetime.now().strftime("%Y%m%d-%H%M%S"))
        early_stopping = EarlyStopping(monitor = 'val_loss',
                                       patience = 3,
                                       restore_best_weights = True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=file_path)
        model.compile(optimizer = optimizer,
                      loss = 'mean_absolute_error')
        self.model = model
        self.early_stopping = early_stopping
        self.tensorboard_callback = tensorboard_callback

    def fit(self, epochs=50, batch_size=40):
        """
        Fits the model and logs all model configurations.
        
        """
        pp = pprint.PrettyPrinter(indent=4)
        formatted_config = pp.pformat(self.model.get_config())
        formatted_opt_config = pp.pformat(self.model.optimizer.get_config())
        self.rnn_logger.info(f"Training {self.variable_name} LSTM Model with configurations:\n{formatted_config}\n{formatted_opt_config}")
        self.history = self.model.fit(
                                    x=[self.X_train, self.X_train],
                                    # x = [self.X_train_forward, self.X_train_backward, self.X_train_forward, self.X_train_backward],
                                    y=self.y_train,
                                    epochs=epochs,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    # validation_data=([self.X_val_forward, self.X_val_backward], self.y_val),
                                    validation_data = ([self.X_val, self.X_val], self.y_val),
                                    verbose=1,
                                    callbacks=[self.early_stopping, self.tensorboard_callback])
        history_df = pd.DataFrame(self.history.history)
        self.rnn_logger.info(f"Parameters:\n{self.history.params}")
        self.rnn_logger.info(f"History:\n{history_df}")
        return self.history

    def predict_bidrectional(self, data_forward, data_backward):
        """
        Method for forecasting / predicting bidirectional model outputs given inputs.
        
        """
        data_forward = np.array([data_forward])
        data_backward = np.array([data_backward])
        assert data_forward.shape[1] == self.X_train_forward.shape[1], "Input data_forward shape mismatch"
        assert data_backward.shape[1] == self.X_train_backward.shape[1], "Input data_backward shape mismatch"
        data_forward = self.scale_data(data_forward)
        data_backward = self.scale_data(data_backward)
        X_forward = data_forward.reshape(data_forward.shape[0], self.X_train_forward.shape[1], self.X_train_forward.shape[2])
        X_backward = data_backward.reshape(data_backward.shape[0], self.X_train_backward.shape[1], self.X_train_backward.shape[2])
        predictions = self.model.predict([X_forward, X_backward])
        predictions = self.invert_scale(predictions)
        return predictions
    
    def predict(self, input_data):
        """
        Method for predicting / forecasting forward model inputs / outputs.
        
        """
        input_data = np.array([input_data])
        X = self.scale_data(input_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        predicted = self.model.predict([X, X])
        predicted_unscaled = self.invert_scale(predicted)
        return predicted_unscaled

    def save_model(self):
        if input("Save model (y/n)? ").lower() == 'n': return
        file_name = f'RNN_Model_{self.variable_name}.h5'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file_name)
        self.model.save(file_path)
        self.rnn_logger.info(f"Saving RNN Model as {file_name}.")

    def load_model(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'RNN_Model_{self.variable_name}.h5')
        self.model = tf.keras.models.load_model(file_path)

    def scale_data(self, data):
        min_val = DATA_SUMMARY_DICT[self.variable_name]['min']
        max_val = DATA_SUMMARY_DICT[self.variable_name]['max']
        scaled_data = 1 + ( (data - min_val) * (99 / (max_val - min_val)) )
        return scaled_data
    
    def invert_scale(self, data):
        min_val = DATA_SUMMARY_DICT[self.variable_name]['min']
        max_val = DATA_SUMMARY_DICT[self.variable_name]['max']
        unscaled_data = min_val + ((data - 1) * (max_val - min_val) / 99)
        return unscaled_data

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def plot_model(self):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', f'rnn_{self.variable_name}_model.png')
        tf.keras.utils.plot_model(
            self.model,
            to_file=file_path,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=200,
            show_layer_activations=True,
            # show_trainable=True ## doesnt work for some
        )


class ModelConfig:
    def __init__(self):
        pass

    def build_model_v5(self, input_shape, predicted_steps):
        print(input_shape, predicted_steps)
        latent_dim = 50

        # Define encoder input
        encoder_input = Input(shape=input_shape)

        # Encoder LSTM layer
        encoder_lstm = LSTM(units=latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_input)
        encoder_states = [state_h, state_c]

        # Define decoder input
        decoder_input = Input(shape=input_shape)

        # Decoder LSTM layer
        decoder_lstm = LSTM(units=latent_dim, return_sequences=False, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)

        # Dense layer for output
        decoder_dense = Dense(units = predicted_steps, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        model = Model([encoder_input, decoder_input], decoder_outputs)

        return model




if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'rnn_log.log')
    logging.basicConfig(filename=file_path,
                        level=logging.INFO,
                        format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info(f"Starting main script...")

    variable = 'WindSpeed'

    file_name = f'{variable}_training_data_15t.json'
    file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model = RNN(input_data = data,
                variable_name = variable)
    model.compile()
    model.fit()
    model.plot_loss()
    model.save_model()
    model.plot_model()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("Complete.")
    logger.info(f"Total runtime: {PVModules.time_converter(total_time)}")
    logger.info("\n# ====================================================================== #\n"
                 + "# =============================== New Run ============================== #\n"
                 + "# ====================================================================== #")