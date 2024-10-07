import unittest
from src.model_evaluation import evaluate_model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        # Generate a synthetic dataset for classification
        self.X, self.y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.model = LogisticRegression().fit(self.X, self.y)

    def test_evaluate_model(self):
        evaluation_result = evaluate_model(self.model, self.X, self.y)
        self.assertIsInstance(evaluation_result, dict, "Evaluation result should be a dictionary.")
        self.assertIn('accuracy', evaluation_result, "Accuracy score is missing from evaluation.")

if __name__ == '__main__':
    unittest.main()
