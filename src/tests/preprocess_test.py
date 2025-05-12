import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing.preprocess import Preprocessing

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'numeric_col': [1, 2, 3, 1000, 5],
            'categorical_col': ['A', 'B', 'A', 'B', 'C'],
            'missing_col': [None, None, None, None, None],
            'target_col': [10, 20, 30, 40, 50]
        }
        self.df = pd.DataFrame(data)
        self.df.to_csv('test.csv', index=False)
        self.preprocessor = Preprocessing('test.csv')

    def tearDown(self):
        import os
        os.remove('test.csv')

    def test_show_stats(self):
        # Ensure no exceptions are raised during execution
        try:
            self.preprocessor.show_stats()
        except Exception as e:
            self.fail(f"show_stats raised an exception: {e}")

    def test_drop_columns_with_all_missing(self):
        self.preprocessor.drop_columns_with_all_missing()
        self.assertNotIn('missing_col', self.preprocessor.df.columns)

    def test_find_outliers(self):
        outliers_dict, outlier_percentages = self.preprocessor.find_outliers()
        self.assertIn('numeric_col', outliers_dict)
        self.assertGreater(outlier_percentages['numeric_col'], 0)

    def test_split_X_Y(self):
        dfs_with_y = self.preprocessor.split_X_Y()
        self.assertEqual(len(dfs_with_y), 1)  # Only one target column
        self.assertIn('target_col', dfs_with_y[0].columns)

    def test_process_X(self):
        self.preprocessor.split_X_Y()
        self.preprocessor.process_X(method='median')
        # Ensure no outliers remain in numeric_col
        outliers_dict, _ = self.preprocessor.find_outliers(self.preprocessor.X)
        self.assertEqual(len(outliers_dict['numeric_col']), 0)

if __name__ == '__main__':
    unittest.main()
