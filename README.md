
# Kernel SVM - Social Network Ads Purchase Prediction

## Project Overview

This project uses a Kernel Support Vector Machine (SVM) classification algorithm to predict whether a customer will purchase a product based on their age and estimated salary.

The project implements a complete Machine Learning pipeline, including data preprocessing, exploratory data analysis, model training, cross-validation, hyperparameter tuning, evaluation, and deployment through an API and web interface.

## Problem Statement

Predict customer purchasing behavior using demographic and salary information.

### Features

- Age
- Estimated Salary

### Target

- Purchased
  - 0: Not purchased
  - 1: Purchased

## Machine Learning Workflow

1. Data loading
2. Exploratory Data Analysis
3. Data preprocessing
4. Train-test splitting
5. Feature scaling
6. Kernel SVM model training
7. K-Fold Cross Validation
8. Hyperparameter optimization
9. Model evaluation
10. Model serialization

## Model

- Algorithm: Support Vector Machine
- Kernel: RBF (Radial Basis Function)
- Feature Scaling: StandardScaler
- Hyperparameter Tuning: GridSearchCV

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC
- Cross-Validation Score

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- FastAPI
- Streamlit
- Docker
- Pytest
- GitHub Actions

## Application Architecture

User → Streamlit Frontend → FastAPI Backend → SVM Model → Prediction

## Project Structure

```text
Kernel-SVM-KFold-CrossValidation/
│
├── main.py
├── api.py
├── app.py
├── Dockerfile
├── compose.yml
├── requirements.txt
├── models/
├── outputs/
├── tests/
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

## Model Performance

- Test Accuracy: 89%
- ROC-AUC Score: 0.9544

*Performance metrics are based on the current experimental run and may vary depending on the training configuration.*

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yashwanth86868/Cross-Validation-complete-ML-project.git
```

### 2. Navigate to the Project

```bash
cd Cross-Validation-complete-ML-project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline

```bash
python main.py
```

### 5. Start the API

```bash
uvicorn api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Frontend

```bash
streamlit run app.py
```

## Run with Docker

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:8501
```

API:

```text
http://localhost:8000
```

## Automated Testing

Run the test suite:

```bash
python -m pytest -v
```

The project includes API tests for:

- Home endpoint
- Health endpoint
- Prediction endpoint
- Invalid age validation
- Negative salary validation

## Continuous Integration

GitHub Actions automatically installs the project dependencies and runs the automated tests whenever changes are pushed or a pull request is created.

## Future Improvements

- Deploy the application to a cloud platform
- Add more customer features
- Improve model performance
- Add monitoring and logging
- Implement authentication
- Explore alternative ML algorithms

## Author

Yashwanth

GitHub: https://github.com/yashwanth86868
