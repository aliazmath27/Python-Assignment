"""Unit Tests for Assignment Module

This module contains unit tests for functions in the Assignment module, specifically the `mapping_points` and
`compare_individual_plots` functions. It ensures the correct functionality of these functions using sample data.

Classes:
    - `TestAssignment`: Test case class for Assignment module functions.

Test Methods:
    - `test_mapping_points`: Tests the `mapping_points` function with a sample test point.
    - `test_compare_individual_plots`: Tests the `compare_individual_plots` function with sample dataframes.

Usage Example:
    # Run the unit tests
    if __name__ == '__main__':
        unittest.main()
"""


import unittest
import pandas as pd
from unittest.mock import patch
from Assignment import mapping_points, compare_individual_plots 
from SqlPackage import MakeConnection, IdealFunc

class TestAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up any necessary data or configurations
        IdealFuncPath = r"Dataset2\ideal.csv"
        TrainPath = r"C:\Users\aliaz\IU\Dataset2\train.csv"
        TestPath = r"C:\Users\aliaz\IU\Dataset2\test.csv"

        cls.ideal_dataset = MakeConnection(path=IdealFuncPath)
        cls.train_dataset = MakeConnection(path=TrainPath)
        cls.test_dataset = MakeConnection(path=TestPath)

    def test_mapping_points(self):
        # Create a sample test point
        test_point = [1, 2]

        with patch('builtins.input', side_effect=[test_point[0]]):
            # Call the mapping_points function
            result = mapping_points(test_point, self.ideal_dataset.data)

        # Add assertions to check the correctness of the result
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        # Add more specific assertions based on your expectations

    def test_compare_individual_plots(self):
        # Create sample dataframes for testing
        df1 = pd.DataFrame({'x': [1, 2, 3], 'y1': [2, 4, 6], 'y2': [3, 6, 9]})
        df2 = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 3, 5], 'y2': [2, 4, 6]})

        with patch('matplotlib.pyplot.show', return_value=None):
            # Call the compare_individual_plots function
            compare_individual_plots(df1, df2)
            # Manual inspection is required for visualization tests

if __name__ == '__main__':
    unittest.main()
    
    