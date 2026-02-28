LAB 1 – MLOps (IE-7374)
Objective

This lab demonstrates a complete MLOps workflow including:

Virtual environment setup

GitHub repository creation

Structured project organization

Unit testing using Pytest and Unittest

Continuous Integration using GitHub Actions

The original calculator example was modified to implement a real Machine Learning workflow using the Titanic dataset.

Modifications Made

Instead of basic arithmetic functions, the project was redesigned to:

Load and preprocess the Titanic dataset

Train a Logistic Regression model

Evaluate model performance using accuracy

Add prediction functionality

Implement automated testing

Enable CI validation using GitHub Actions

Dataset Used

Titanic Dataset (Titanic-Dataset.csv)

Selected Features:

Pclass

Sex

Age

Fare

Target:

Survived

Model Logic

Load dataset using pandas.

Convert categorical feature Sex to numeric.

Handle missing values in Age using median imputation.

Split data into training and test sets.

Train Logistic Regression model.

Evaluate accuracy on test set.

Assert accuracy > 0.6 in unit tests.

Testing

Two testing frameworks were implemented:

Pytest

Validates data loading

Validates model training

Ensures minimum accuracy threshold

Unittest

Class-based test validation

Confirms training and performance logic

Continuous Integration

GitHub Actions workflows:

Automatically install dependencies

Run Pytest

Run Unittest

Fail if tests do not pass

All tests must pass before merging into main branch.
