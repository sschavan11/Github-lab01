import unittest
from src.titanic_model import load_data, train_model


class TestTitanicModel(unittest.TestCase):

    def test_training_with_path(self):
        df = load_data("data/Titanic-Dataset.csv")
        model, accuracy = train_model(df)
        self.assertGreater(accuracy, 0.6)

    def test_training_with_defaults(self):
        df = load_data()
        model, accuracy = train_model(df)
        self.assertGreater(accuracy, 0.6)


if __name__ == "__main__":
    unittest.main()
