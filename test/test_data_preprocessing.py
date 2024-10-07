import unittest
import pandas as pd
from src.data_preprocessing import handle_missing_values, encode_categorical_columns, split_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'Amount': [100, 200, None, 400],
            'Category': ['A', 'B', 'A', 'C']
        })
        self.target_column = 'Amount'
    
    def test_handle_missing_values(self):
        df_filled = handle_missing_values(self.df)
        self.assertFalse(df_filled.isnull().values.any(), "Missing values should be handled.")
    
    def test_encode_categorical_columns(self):
        categorical_columns = ['Category']
        df_encoded = encode_categorical_columns(self.df, categorical_columns)
        self.assertTrue('Category_A' in df_encoded.columns, "Category encoding failed.")
    
    def test_split_data(self):
        df_filled = handle_missing_values(self.df)
        X_train, X_test, y_train, y_test = split_data(df_filled, self.target_column)
        self.assertEqual(len(X_train), 3, "Train set size is incorrect.")
        self.assertEqual(len(X_test), 1, "Test set size is incorrect.")

if __name__ == '__main__':
    unittest.main()
