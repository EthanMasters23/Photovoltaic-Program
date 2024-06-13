#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: on May 28 11:12 2023

@author: Ethan Masters

Purpose: PV Project GUI, used to run overhead programs.

Python Version: Python 3.10.11 
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QFrame, QLabel, QGroupBox, QGridLayout, QVBoxLayout,
    QComboBox, QHBoxLayout, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from pprint import pprint

from data_compiler import CompilerPipeline
from data_summary import SummaryPipeline
from data_imputer import ImputerPipeline
from imputation_method_testing import MethodTestingPipeline
from nan_data_summary import NaNSummaryPipeline
from nan_data_summary_table import NaNSummaryTablePipeline
from feature_engineering import FeatureEngineeringPipeline
from pv_constants import DATA_FILE_OPTIONS, MONTH_DICT


class PV_Solar_GUI(QWidget):
    """
    PV Gui Application

    Graphical user interface to modularlize and provide
    a user interface for running and visualizing PV Project Programs.

    Methods: 
        (None)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PV-Solar Data Imputation Project")
        self.setMinimumSize(900, 600)
        self.init_ui()

        # self.worker = Worker()
        # self.worker.error_encountered.connect(self.display_error)

    def init_ui(self):
        self.setup_title_label()
        self.setup_main_actions_group()
        self.setup_sub_actions_group()
        self.setup_layout()

    def setup_title_label(self):
        self.label_title = QLabel("PV Solar Program", alignment = Qt.AlignCenter)
        self.label_title.setFont(QFont("Arial", 16))
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)

    def setup_main_actions_group(self):
        title_label = QLabel("Main Programs")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QFont("Arial", 14)
        title_label.setFont(title_font)
        self.group_main_actions = QGroupBox()
        layout_main_actions = QGridLayout(self.group_main_actions)
        layout_main_actions.addWidget(title_label, 0, 0, 1, -1)
        self.setup_main_buttons(layout_main_actions)
        self.setup_back_button()

    def setup_back_button(self):
        self.button_back = QPushButton("Back")
        self.button_back.clicked.connect(self.show_main_actions)
        self.button_back.hide()

    def setup_main_buttons(self, layout_main_actions):
        buttons = {
            "Data Imputer" : self.create_button("Data Imputer"),
            "Data Imputation Testing" : self.create_button("Data Imputation Testing"),
            "Feature Engineering" : self.create_button("Feature Engineering"),
            "Data Summary" : self.create_button("Data Summary"),
            "Data Compiler" : self.create_button("Data Compiler"),
            "NaN Data Summary" : self.create_button("NaN Data Summary"),
            "NaN Data Summary Table" : self.create_button("NaN Data Summary Table")
        }
        for i, (text, button) in enumerate(buttons.items()):
            button.clicked.connect(self.show_sub_actions)
            i+=1
            layout_main_actions.addWidget(button,
                                        i if i % 2 else i - 1,
                                        0 if i % 2 else 1)

    def create_button(self, text):
        button = QPushButton(text)
        button.setMinimumWidth(150)
        return button

    def setup_sub_actions_group(self):
        self.group_sub_actions = QGroupBox()
        self.group_sub_actions.hide()

    def setup_layout(self):
        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.label_title, 0, 0, 1, -1)
        self.main_layout.addWidget(self.separator, 1, 0, 1, -1)
        self.main_layout.addWidget(self.group_main_actions, 2, 0, 1, -1)
        self.main_layout.addWidget(self.group_sub_actions, 3, 0, 1, -1)
        self.main_layout.addWidget(self.button_back, 4, 0, 1, -1)
        self.setLayout(self.main_layout)

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################

    def show_main_actions(self):
        self.group_main_actions.show()
        self.group_sub_actions.hide()
        self.button_back.hide()

    def show_sub_actions(self):
        self.group_main_actions.hide()
        sender_text = self.sender().text()
        self.group_sub_actions.deleteLater()
        self.group_sub_actions = QGroupBox()
        self.main_layout.addWidget(self.group_sub_actions, 3, 0, 1, -1)
        layout_sub_actions = QHBoxLayout(self.group_sub_actions)
        if sender_text == "Data Summary":
            self.add_data_summary_sub_actions(layout_sub_actions)
        elif sender_text == "Data Compiler":
            self.add_data_compiler_sub_actions(layout_sub_actions)
        elif sender_text == "Data Imputer":
            self.add_data_imputer_sub_actions(layout_sub_actions)
        elif sender_text == "Data Imputation Testing":
            self.add_data_imputation_testing_sub_actions(layout_sub_actions)
        elif sender_text == "NaN Data Summary":
            self.add_nan_data_summary_sub_actions(layout_sub_actions)
        elif sender_text == "NaN Data Summary Table":
            self.add_nan_data_summary_table_sub_actions(layout_sub_actions)
        elif sender_text == "Feature Engineering":
            self.add_feature_engineering_sub_action(layout_sub_actions)

        self.group_sub_actions.setLayout(layout_sub_actions)
        self.group_sub_actions.adjustSize()
        self.group_sub_actions.show()
        self.button_back.show()

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################

    def add_data_summary_sub_actions(self, layout):
        run_month_layout = QGridLayout()
        self.set_title_sub_action(run_month_layout, "Run Month Data Summary Program")
        self.run_month_file_type = QComboBox()
        self.collect_file_type(run_month_layout, self.run_month_file_type)
        self.collect_year_month(run_month_layout)
        self.add_run_button(run_month_layout, self.run_month_summary_data_func)
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run Data Summary Program")
        self.run_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_file_type)
        self.add_run_button(run_layout, self.run_summary_data_func)
        load_layout = QGridLayout()
        self.set_title_sub_action(load_layout, "Load Data Summary Program")
        self.load_file_type = QComboBox()
        self.collect_file_type(load_layout, self.load_file_type)
        self.add_load_button(load_layout, self.load_data_summary_func)
        layout.addLayout(run_month_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(run_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(load_layout)

    def add_data_compiler_sub_actions(self, layout):
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run Data Compiler Program")
        self.collect_file_option(run_layout)
        self.run_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_file_type)
        self.add_run_button(run_layout, self.run_data_compiler_func)
        load_layout = QGridLayout()
        self.set_title_sub_action(load_layout, "Load Data Compiler Program")
        self.collect_file_option(load_layout)
        self.load_file_type = QComboBox()
        self.collect_file_type(load_layout, self.load_file_type)
        self.add_load_button(load_layout, self.load_data_compiler_func)
        layout.addLayout(run_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(load_layout)

    def add_data_imputer_sub_actions(self, layout):
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run Data Imputer Program")
        self.run_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_file_type)
        self.add_run_button(run_layout, self.run_data_imputer_func)
        run_month_layout = QGridLayout()
        self.set_title_sub_action(run_month_layout, "Run Month Data Imputer Program")
        self.run_month_file_type = QComboBox()
        self.collect_file_type(run_month_layout, self.run_month_file_type)
        self.collect_year_month(run_month_layout)
        self.add_run_button(run_month_layout, self.run_month_data_imputer_func)
        layout.addLayout(run_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(run_month_layout)

    def add_data_imputation_testing_sub_actions(self, layout):
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run Data Imputation Testing Program")
        self.run_month_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_month_file_type)
        self.collect_year_month(run_layout)
        self.collect_gap_size(run_layout)
        self.collect_testing_iterations(run_layout)
        self.add_run_button(run_layout, self.run_imputation_method_testing_func)
        layout.addLayout(run_layout)

    def add_feature_engineering_sub_action(self, layout):
        feature_layout = QVBoxLayout()
        title_label = QLabel("Run Feature Engineering Program")
        title_label.setAlignment(QtCore.Qt.AlignCenter) 
        title_font = QFont("Arial", 14)
        title_label.setFont(title_font)
        feature_layout.addWidget(title_label)
        self.feature_file_type = QComboBox()
        self.collect_file_type(feature_layout, self.feature_file_type)
        self.add_feature_checkbox(feature_layout)
        self.add_run_button(feature_layout, self.run_feature_engineering_func)
        load_layout = QGridLayout()
        self.set_title_sub_action(load_layout, "Load Feature Engineering Program")
        self.load_file_type = QComboBox()
        self.collect_file_type(load_layout, self.load_file_type)
        self.add_load_button(load_layout, self.load_feature_engineering_func)
        graph_layout = QGridLayout()
        self.set_title_sub_action(graph_layout, "Graph Feature Engineering Program")
        self.graph_file_type = QComboBox()
        self.collect_file_type(graph_layout, self.graph_file_type)
        self.add_feature_graphs(graph_layout)
        self.add_run_button(graph_layout, self.run_feature_engineering_graph_func)
        layout.addLayout(feature_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(load_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(graph_layout)

    def add_nan_data_summary_sub_actions(self, layout):
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run NaN Data Summary Program")
        self.run_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_file_type)
        self.add_run_button(run_layout, self.run_nan_summary_func)
        run_month_layout = QGridLayout()
        self.set_title_sub_action(run_month_layout, "Run Month NaN Data Summary Program")
        self.run_month_file_type = QComboBox()
        self.collect_file_type(run_month_layout, self.run_month_file_type)
        self.collect_year_month(run_month_layout)
        self.add_run_button(run_month_layout, self.run_month_nan_summary_func)
        layout.addLayout(run_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(run_month_layout)

    def add_nan_data_summary_table_sub_actions(self, layout):
        run_layout = QGridLayout()
        self.set_title_sub_action(run_layout, "Run NaN Data Summary Table Program")
        self.run_file_type = QComboBox()
        self.collect_file_type(run_layout, self.run_file_type)
        self.add_run_button(run_layout, self.run_nan_summary_table_func)
        load_layout = QGridLayout()
        self.set_title_sub_action(load_layout, "Load NaN Data Summary Table Program")
        self.load_file_type = QComboBox()
        self.collect_file_type(load_layout, self.load_file_type)
        self.add_load_button(load_layout, self.load_nan_summary_table_func)
        layout.addLayout(run_layout)
        self.add_vline_seperator(layout)
        layout.addLayout(load_layout)

    def set_title_sub_action(self, layout, title):
        title_label = QLabel(title)
        title_label.setAlignment(QtCore.Qt.AlignCenter) 
        title_font = QFont("Arial", 14)
        title_label.setFont(title_font)
        layout.addWidget(title_label, 0, 0, 1, -1)

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################

    def add_load_button(self, layout, func):
        button_submit = QPushButton("Load")
        layout.addWidget(button_submit)
        button_submit.clicked.connect(func)

    def add_run_button(self, layout, func):
        button_submit = QPushButton("Run")
        layout.addWidget(button_submit)
        button_submit.clicked.connect(func)

    def add_vline_seperator(self, layout):
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
    
    def add_hline_seperator(self, layout):
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

    def add_feature_checkbox(self, layout):
        overlay = QVBoxLayout()
        generate_features = QGridLayout()
        self.checkbox1 = QCheckBox('Generate Solar Zenith Feature')
        self.checkbox2 = QCheckBox('Generate Seasonal RBF')
        self.checkbox3 = QCheckBox('Generate Cos and Sin Transformations')
        self.checkbox4 = QCheckBox('Generate Training Data')
        format_features = QGridLayout()
        self.checkbox5 = QCheckBox('Remove Night Values')
        self.checkbox6 = QCheckBox('Set Night Values')
        self.checkbox7 = QCheckBox('Remove NaN Values')
        generate_features.addWidget(self.checkbox1)
        generate_features.addWidget(self.checkbox2)
        generate_features.addWidget(self.checkbox3)
        generate_features.addWidget(self.checkbox4)
        format_features.addWidget(self.checkbox5)
        format_features.addWidget(self.checkbox6)
        format_features.addWidget(self.checkbox7)
        overlay.addLayout(generate_features)
        self.add_hline_seperator(overlay)
        overlay.addLayout(format_features)
        layout.addLayout(overlay)

    def add_feature_graphs(self, layout):
        self.checkbox8 = QCheckBox('Graph RBF Graphs')
        self.checkbox9 = QCheckBox('Graph Cos and and Graphs')
        layout.addWidget(self.checkbox8)
        layout.addWidget(self.checkbox9)

    def collect_file_option(self, layout):
        self.combo_file_option = QComboBox()
        self.combo_file_option.addItems(["processed", "raw"])
        layout.addWidget(QLabel("Select FILE_OPTION:"))
        layout.addWidget(self.combo_file_option)

    def collect_file_type(self, layout, file_type):
        file_type.addItems(["Irradiance", "Deger", "Fixed"])
        file_type.setCurrentText("Irradiance")
        layout.addWidget(QLabel("Select FILE_TYPE:"))
        layout.addWidget(file_type)
    
    def collect_gap_size(self, layout):
        self.spin_gap_size = QSpinBox()
        self.spin_gap_size.setMinimum(1)
        self.spin_gap_size.setMaximum(100000)
        self.spin_gap_size.setValue(10)
        layout.addWidget(QLabel("Set Gap Size (Seconds):"))
        layout.addWidget(self.spin_gap_size)

    def collect_testing_iterations(self, layout):
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setMinimum(1)
        self.spin_iterations.setMaximum(100)
        self.spin_iterations.setValue(5)
        layout.addWidget(QLabel("Set Number of Iterations:"))
        layout.addWidget(self.spin_iterations)

    def collect_year_month(self, layout):
        self.year_combobox = QComboBox()
        self.year_combobox.addItems(DATA_FILE_OPTIONS["years"])
        self.month_combobox = QComboBox()
        self.month_combobox.addItems(DATA_FILE_OPTIONS["months"])

        self.year_combobox.setCurrentText("2022")
        self.month_combobox.setCurrentText("July")

        layout.addWidget(QLabel("Select YEAR:"))
        layout.addWidget(self.year_combobox)
        layout.addWidget(QLabel("Select MONTH:"))
        layout.addWidget(self.month_combobox)

   

    ####################################################################
    ####################################################################
    ####################################################################
    ####################################################################

    def run_summary_data_func(self):
        print("Starting Data Summary Program...")
        file_type = self.run_file_type.currentText()
        SummaryPipeline(
            FILE_TYPE = file_type,
        ).run()
        print("Finished.")

    def run_month_summary_data_func(self):
        print("Starting Month Data Summary Program...")
        file_type = self.run_month_file_type.currentText()
        year = self.year_combobox.currentText()
        month = MONTH_DICT[self.month_combobox.currentText()]
        SummaryPipeline(
            FILE_TYPE = file_type,
            YEAR = year,
            MONTH = month
        ).run()
        print("Finished.")

    def load_data_summary_func(self):
        print("Loading Data Summary Program...")
        file_type = self.load_file_type.currentText()
        df = SummaryPipeline(
            FILE_TYPE = file_type
        ).load()
        print("Finshed.")

    def run_data_compiler_func(self):
        print("Starting Data Compiler Program...")
        file_type = self.run_file_type.currentText()
        file_option = self.combo_file_option.currentText()
        CompilerPipeline(
            FILE_OPTION = file_option,
            FILE_TYPE = file_type
        ).run()
        print("Finished.")

    def load_data_compiler_func(self):
        print("Loading Data Compiler Program...")
        file_option = self.combo_file_option.currentText()
        file_type = self.load_file_type.currentText()
        caller = CompilerPipeline(
            FILE_OPTION = file_option,
            FILE_TYPE = file_type
        ).load()
        pprint(caller)
        print("Finished.")

    def run_feature_engineering_func(self):
        print("Starting Feature Engineering Program...")
        caller = FeatureEngineeringPipeline().feature_engineer
        if self.checkbox1.isChecked():
            caller.generate_solar_zenith_feature()
        if self.checkbox2.isChecked():
            caller.generate_time_features_rbf()
        if self.checkbox3.isChecked():
            caller.generate_time_features_cos_sin_trans()
        if self.checkbox5.isChecked():
            caller.remove_night()
        if self.checkbox6.isChecked():
            caller.set_night()
        if self.checkbox4.isChecked():
            caller.generate_clean_data()
        if self.checkbox7.isChecked():
            caller.remove_nan_rows()

        caller.save_clean_training_data()
        caller.save_features_data_frame()
        caller.save_training_data_frame()
        print("Finished.")

    def run_feature_engineering_graph_func(self):
        print("Starting Feature Engineering Grapher...")
        caller = FeatureEngineeringPipeline().feature_engineer
        if self.checkbox1.isChecked():
            caller.graph_time_features_cos_sin_trans()
        if self.checkbox2.isChecked():
            caller.graph_time_features_rbf()
        print("Finished.")

    def load_feature_engineering_func(self):
        print("Loading Feature Engineering Program...")
        caller = FeatureEngineeringPipeline().feature_engineer
        if self.checkbox1.isChecked():
            caller.graph_time_features_rbf()
        if self.checkbox2.isChecked():
            caller.graph_time_features_cos_sin_trans()
        print("Finished.")

    def run_data_imputer_func(self):
        print("Starting Data Imputer Program...")
        file_type = self.run_file_type.currentText()
        ImputerPipeline(
            FILE_TYPE = file_type,
        ).run()
        print("Finished.")

    def run_month_data_imputer_func(self):
        print("Starting Month Data Imputer Program...")
        file_type = self.run_month_file_type.currentText()
        year = self.year_combobox.currentText()
        month = MONTH_DICT[self.month_combobox.currentText()]
        ImputerPipeline(
            FILE_TYPE = file_type,
            YEAR = year,
            MONTH = month
        ).run()
        print("Finished.")

    def run_imputation_method_testing_func(self):
        print("Starting Imputation Method Testing Program...")
        gap_size = self.spin_gap_size.value()
        interations = self.spin_iterations.value()
        file_type = self.run_month_file_type.currentText()
        year = self.year_combobox.currentText()
        month = MONTH_DICT[self.month_combobox.currentText()]
        MethodTestingPipeline(
            GAP_LENGTH = gap_size,
            ITERATIONS = interations,
            FILE_TYPE = file_type,
            YEAR = year,
            MONTH = month
        ).run()
        print("Finished.")

    def run_nan_summary_func(self):
        print("Starting NaN Summary Program...")
        file_type = self.run_file_type.currentText()
        caller = NaNSummaryPipeline(
            FILE_TYPE = file_type
        )
        caller.run()
        print("Finished.")

    def run_month_nan_summary_func(self):
        print("Starting NaN Summary Program...")
        file_type = self.run_month_file_type.currentText()
        year = self.year_combobox.currentText()
        month = MONTH_DICT[self.month_combobox.currentText()]
        caller = NaNSummaryPipeline(
            FILE_TYPE = file_type,
            YEAR = year,
            MONTH = month
        )
        caller.run()
        print("Finished.")

    def run_nan_summary_table_func(self):
        print("Starting NaN Summary Table Program...")
        file_type = self.run_file_type.currentText()
        NaNSummaryTablePipeline(
            FILE_TYPE = file_type
        ).run()
        print("Finished.")

    def load_nan_summary_table_func(self):
        print("Loading NaN Summary Table...")
        file_type = self.load_file_type.currentText()
        caller = NaNSummaryTablePipeline(
            FILE_TYPE = file_type
        )
        caller.load()
        print("Finished.")

    def display_error(self):
        error_message = 'message'
        QMessageBox.critical(self, "Error", error_message)

class Worker(QObject):
    def __init__(self):
        super().__init__()
        self.error_encountered = pyqtSignal()

    def load_data(self):
        try:
            file_path = 'nontin'
            with open(file_path, 'r') as file:
                pass
        except FileNotFoundError as e:
            self.error_encountered.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PV_Solar_GUI()
    window.show()
    sys.exit(app.exec_())