import unittest
from src.model_training import train_logistic_regression, train_random_forest, train_gradient_boosting
from sklearn.datasets import make_classification

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Generate a synthetic dataset for classification
        self.X, self.y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    def test_train_logistic_regression(self):
        model = train_logistic_regression(self.X, self.y)
        self.assertIsNotNone(model, "Logistic Regression training failed.")
    
    def test_train_random_forest(self):
        model = train_random_forest(self.X, self.y)
        self.assertIsNotNone(model, "Random Forest training failed.")
    
    def test_train_gradient_boosting(self):
        model = train_gradient_boosting(self.X, self.y)
        self.assertIsNotNone(model, "Gradient Boosting training failed.")

if __name__ == '__main__':
    unittest.main()
