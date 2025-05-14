# 🌸 SVM Classifier on Iris Dataset – Interview Ready Guide

This guide demonstrates how to train a **Support Vector Machine (SVM)** classifier using the classic **Iris dataset**. It includes step-by-step explanations and common interview Q&A.

---

## 🚀 Steps Summary – LSTEPS Mnemonic

| Step | Description           | Code Example                      |
|------|------------------------|------------------------------------|
| **L** | Load the dataset       | `datasets.load_iris()`             |
| **S** | Split into train/test | `train_test_split(...)`            |
| **T** | Transform (scale)     | `StandardScaler()`                 |
| **E** | Estimate (train model)| `SVC(kernel='linear').fit(...)`    |
| **P** | Predict                | `model.predict(...)`               |
| **S** | Score (evaluate)      | `accuracy_score(...)`              |

---

## 📌 Full Code Example

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train SVM model
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# 5. Predict and Evaluate
predictions = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# 6. Predict on new samples
new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]
scaled_new_data = scaler.transform(new_data)
new_predictions = svm_classifier.predict(scaled_new_data)
print(f"Predictions for new data: {new_predictions}")
