import unittest
from src.utils import load_data, feature_engineering
import pandas as pd

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'Total_Transaction_Amount': [1000, 500, 2000],
            'Transaction_Count': [10, 5, 20]
        })
    
    def test_load_data(self):
        try:
            df = load_data('data/sample_data.csv')
            self.assertIsInstance(df, pd.DataFrame, "Loaded data should be a DataFrame.")
        except FileNotFoundError:
            self.skipTest("sample_data.csv not found.")

    def test_feature_engineering(self):
        df_transformed = feature_engineering(self.df)
        self.assertIn('Transaction_Ratio', df_transformed.columns, "Feature engineering failed to create 'Transaction_Ratio'.")

if __name__ == '__main__':
    unittest.main()
