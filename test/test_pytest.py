import pytest
from src.titanic_model import load_data, train_model


def test_training_with_path():
    # explicit path should work (use the actual file name in the repo)
    df = load_data("data/Titanic-Dataset.csv")
    model, accuracy = train_model(df)
    assert accuracy > 0.6


def test_training_with_defaults():
    # functions accept no arguments and use the bundled dataset
    df = load_data()
    model, accuracy = train_model(df)
    assert accuracy > 0.6
