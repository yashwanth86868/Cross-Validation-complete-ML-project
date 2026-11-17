
# ============================================================
# KERNEL SVM MACHINE LEARNING PROJECT
# Social Network Ads Classification
# ============================================================

# 1. IMPORT LIBRARIES

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# 3. IMPORT DATASET
# ============================================================

dataset = pd.read_csv(r"C:\Users\ybali\Downloads\Social_Network_Ads.csv")

print("\nDataset Loaded Successfully!")

print(dataset.head())

print("\nDataset Information:")
print(dataset.info())

print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nStatistical Summary:")
print(dataset.describe())

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(data=dataset, x="Purchased")

plt.title("Purchase Distribution")
plt.savefig("outputs/purchase_distribution.png")
plt.show()

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=dataset,
    x="Age",
    y="EstimatedSalary",
    hue="Purchased"
)

plt.title("Age vs Estimated Salary")
plt.savefig("outputs/feature_distribution.png")
plt.show()

# ============================================================
# 5. FEATURE SELECTION
# ============================================================

X = dataset[["Age", "EstimatedSalary"]].values

y = dataset["Purchased"].values

# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ============================================================
# 7. CREATE PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", random_state=0))
])

# ============================================================
# 8. TRAIN MODEL
# ============================================================

pipeline.fit(X_train, y_train)

print("\nInitial SVM Model Trained Successfully!")

# ============================================================
# 9. PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("SVM Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# ============================================================
# 11. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\nMODEL PERFORMANCE")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

# ============================================================
# 12. TRAINING AND TESTING ACCURACY
# ============================================================

train_accuracy = pipeline.score(X_train, y_train)

test_accuracy = pipeline.score(X_test, y_test)

print("\nTraining Accuracy:", train_accuracy)

print("Testing Accuracy:", test_accuracy)

# ============================================================
# 13. ROC CURVE AND AUC
# ============================================================

y_scores = pipeline.decision_function(X_test)

auc = roc_auc_score(y_test, y_scores)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_scores
)

print("\nAUC Score:", auc)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"SVM AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("outputs/roc_curve.png")

plt.show()

# ============================================================
# 14. K-FOLD CROSS VALIDATION
# ============================================================

kfold_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=10,
    scoring="accuracy"
)

print("\n10-FOLD CROSS VALIDATION")

print("Individual Scores:", kfold_scores)

print("Mean Accuracy:", kfold_scores.mean() * 100)

print("Standard Deviation:", kfold_scores.std() * 100)

# ============================================================
# 15. STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

stratified_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring="accuracy"
)

print("\nSTRATIFIED K-FOLD")

print("Mean Accuracy:",
      stratified_scores.mean() * 100)

print("Standard Deviation:",
      stratified_scores.std() * 100)

# ============================================================
# 16. GRID SEARCH CV
# ============================================================

param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": [0.001, 0.01, 0.1, 1],
    "classifier__kernel": ["rbf"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nGRID SEARCH RESULTS")

print("Best Parameters:", grid_search.best_params_)

print("Best CV Accuracy:", grid_search.best_score_)

# ============================================================
# 17. RANDOMIZED SEARCH CV
# ============================================================

from scipy.stats import loguniform

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={
        "classifier__C": loguniform(0.01, 1000),
        "classifier__gamma": loguniform(0.0001, 10),
        "classifier__kernel": ["rbf"]
    },
    n_iter=20,
    cv=skf,
    scoring="accuracy",
    random_state=0,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nRANDOMIZED SEARCH RESULTS")

print("Best Parameters:",
      random_search.best_params_)

print("Best CV Accuracy:",
      random_search.best_score_)

# ============================================================
# 18. FINAL MODEL SELECTION
# ============================================================

best_model = grid_search.best_estimator_

# ============================================================
# 19. FINAL MODEL EVALUATION
# ============================================================

final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_auc = roc_auc_score(
    y_test,
    best_model.decision_function(X_test)
)

print("\nFINAL MODEL PERFORMANCE")

print("Final Accuracy:", final_accuracy)

print("Final AUC:", final_auc)

print("\nFinal Classification Report:")

print(classification_report(
    y_test,
    final_predictions
))

# ============================================================
# 20. LEARNING CURVE
# ============================================================

train_sizes, train_scores, validation_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=skf,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)

validation_mean = np.mean(validation_scores, axis=1)

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_mean,
    label="Training Score"
)

plt.plot(
    train_sizes,
    validation_mean,
    label="Validation Score"
)

plt.xlabel("Training Examples")

plt.ylabel("Accuracy")

plt.title("Learning Curve")

plt.legend()

plt.savefig("outputs/learning_curve.png")

plt.show()

# ============================================================
# 21. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_svm_model.pkl"
)

print("\nFinal Model Saved Successfully!")

# ============================================================
# 22. LOAD SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    "models/best_svm_model.pkl"
)

# ============================================================
# 23. TEST WITH NEW DATA
# ============================================================

new_customer = np.array([[35, 60000]])

new_prediction = loaded_model.predict(
    new_customer
)

print("\nNEW CUSTOMER PREDICTION")

if new_prediction[0] == 1:

    print("Customer may purchase the product.")

else:

    print("Customer may not purchase the product.")

print("\nPROJECT EXECUTION COMPLETED!")