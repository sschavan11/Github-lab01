# LAB 1 – MLOps (IE-7374)

## Objective

This lab demonstrates a complete MLOps workflow including:

- Virtual environment setup  
- GitHub repository creation  
- Structured project organization  
- Unit testing using Pytest and Unittest  
- Continuous Integration using GitHub Actions  

The original calculator-based example was modified to implement a Machine Learning workflow using the Titanic dataset.

---

## Dataset

**Titanic Dataset (Titanic-Dataset.csv)**

Selected Features:
- Pclass  
- Sex  
- Age  
- Fare  

Target Variable:
- Survived  

---

## Implementation Logic

### 1. Data Loading & Preprocessing
- Loaded dataset using pandas.
- Selected relevant columns.
- Converted categorical variable `Sex` to numerical format.
- Handled missing values in `Age` using median imputation.
- Dropped remaining null values if any.

### 2. Model Training
- Used Logistic Regression from scikit-learn.
- Split dataset into training and testing sets.
- Trained model on training data.
- Evaluated performance using accuracy score.
- Ensured model accuracy is greater than 0.6.

### 3. Testing

Two testing frameworks were implemented:

#### Pytest
- Validates data loading.
- Validates model training.
- Ensures accuracy threshold is met.

#### Unittest
- Class-based test validation.
- Confirms correct model training and performance.

---

## Continuous Integration (CI)

GitHub Actions workflows were configured to:

- Automatically install dependencies.
- Run Pytest.
- Run Unittest.
- Fail the workflow if any test fails.

This ensures code reliability and reproducibility.

## Project Structure
.github/workflows/
data/
src/
test/
README.md
requirements.txt

---
Test images
<img width="1104" height="192" alt="Screenshot 2026-02-28 031548" src="https://github.com/user-attachments/assets/0a0972ef-f293-4bc7-9396-92f78a0fddf8" />

<img width="1904" height="960" alt="Screenshot 2026-02-28 031400" src="https://github.com/user-attachments/assets/bef41952-51f2-49bc-8a5d-1391f8d1c568" />

---

## How to Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
python -m unittest test.test_unittest



