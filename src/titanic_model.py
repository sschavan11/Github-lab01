import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(path=None):
    """Load and preprocess the Titanic dataset.

    Parameters
    ----------
    path : str or None
        Path to the CSV file. If None, a default dataset bundled in the
        repository is used.
    """

    if path is None:
        # build a sensible default relative to this module location
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        path = os.path.join(base_dir, "Titanic-Dataset.csv")

    df = pd.read_csv(path)
    # select only the relevant columns; this produces a new DataFrame but
    # we'll continue working with the returned object directly to avoid
    # chained assignment issues.
    df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].copy()

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # avoid inplace operations on a view: assign filled values back to column
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # as a safety net, drop any remaining rows with NaNs (shouldn't be needed but
    # prevents errors if other columns had missing values)
    df = df.dropna()

    return df


def train_model(df=None):
    """Train a logistic regression model on the Titanic data.

    Parameters
    ----------
    df : pandas.DataFrame or None
        Preloaded dataset. If None the default CSV is read.

    Returns
    -------
    model : sklearn.base.BaseEstimator
        Fitted logistic regression model.
    accuracy : float
        Accuracy on the held-out test set.
    """

    if df is None:
        df = load_data()

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def predict_survival(model, features):
    return model.predict([features])
