"""Module for Data Connection and Ideal Function Calculation

This module provides classes for reading CSV data, creating SQL tables, and calculating the maximum deviation 
between a provided ideal function and a corresponding training function.

Classes:
    - `MakeConnection`: Handles CSV data reading and SQL table creation.
    - `PFunction`: Parent class for ideal functions.
    - `IdealFunc`: Child class that calculates the maximum deviation between a train and an ideal function.

Usage Example:
    # Create a connection to CSV data
    connection = MakeConnection(path="path/to/data.csv")

    # Create an ideal function instance
    ideal_function = IdealFunc(ideal_data, train_data, name="ExampleFunction")

    # Calculate the maximum deviation
    max_deviation = ideal_function.calc_max_deviation()

    # Create an SQL table
    connection.CreatSql(file_name="ExampleDB", title="ExampleTable")
"""




import pandas as pd
from sqlalchemy import create_engine
import numpy as np

class MakeConnection:
    # Read csv
    def __init__(self,path):
        self.dataFrames = []
        try:
            self.data = pd.read_csv(path)
        except FileNotFoundError:
            print("Issue while reading file {}".format(path))
            raise

    def CreatSql(self, file_name, title):

        db_engine = create_engine('sqlite:///{}.db'.format(file_name), echo=False)

        # Using dbEngine and saving data
        data = self.data.copy()
        data.columns = [name.capitalize() + title for name in data.columns]
        data.set_index(data.columns[0], inplace=True)

        data.to_sql(title, db_engine, if_exists="replace", index=True)

class PFunction:

    def __init__(self, ideal, name):
        self._name = name
        self.ideal = ideal


class IdealFunc(PFunction):
    def __init__(self, ideal, train, name):
        self.train_function = train
        super().__init__(ideal, name)

    

    def calc_max_deviation(self):
        deviation = self.train_function - self.ideal

        # Ensure all values in the deviation array are numeric
        deviation_numeric = np.asarray(deviation, dtype=np.float64)

        # Replace NaN values with 0
        deviation_numeric = np.nan_to_num(deviation_numeric)

        # Calculate the max absolute deviation
        max_deviation = np.max(np.abs(deviation_numeric))

        return max_deviation

