import unittest
import pandas as pd
from src.visualization import plot_risk_distribution

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'Credit_Score_Category': ['Low', 'Medium', 'Low', 'High']
        })
    
    def test_plot_risk_distribution(self):
        try:
            plot_risk_distribution(self.df)
        except Exception as e:
            self.fail(f"plot_risk_distribution raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
