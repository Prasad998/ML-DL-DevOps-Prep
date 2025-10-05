My Resources for the preparation of Machine Learning and Deep Learning Interviews
[Shaw Talebi](https://github.com/ShawhinT)  

# [123ofAI -  ML coding Problems](https://products.123ofai.com/qnalab/problems)


### Supervised Learning
---
1. [Kernel SVM using numpy](https://products.123ofai.com/qnalab/problems/kernel-svm-using-numpy)


```
Implement a Support Vector Machine (SVM) classifier using the kernel trick, written from scratch using NumPy.

Given:

A feature matrix X
Binary labels y (must be -1 or 1)
A test point test_x

Your task is to:

Use the RBF kernel to compute the kernel matrix
Solve the dual form of the SVM (with simplified gradient ascent)
Predict the class label of test_x after training

Example 1:
Input: X = np.array([[1, 2], [2, 1]]) y = np.array([1, -1]) test_x = np.array([1.5, 1.5])

Output: 1

Explanation:
-The two points form a simple XOR-style layout in 2D space.
-The test point [1.5, 1.5] lies midway and is closer to the positive class [1, 2].
-The RBF kernel emphasizes locality. Since [1.5, 1.5] is more similar to [1, 2] than to [2, 1], it gets classified as 1.


Example 2:
Input: X = np.array([[1, 2], [3, 4]]) y = np.array([1, -1]) test_x = np.array([2, 3])

Output: -1
Explanation:
-This dataset is linearly separable, and test_x = [2, 3] is exactly midway.
-However, [2, 3] is closer (in RBF distance) to [3, 4], which has label -1.
-The kernel SVM decision boundary is non-linear and will lean toward the closer support vector. So, the model predicts -1.
```
**Solution**
```py
import numpy as np

def kernel_svm_predict(X: np.ndarray, y: np.ndarray, test_x: np.ndarray) -> int:
    X = np.asarray(X)
    y = np.asarray(y)
    test_x = np.asarray(test_x)
    
    n = X.shape[0]
    gamma = 1.0
    alpha = np.zeros(n)
    lr = 0.01
    epochs = 500

    # Compute full kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.sum((X[i] - X[j])**2) # 
            K[i, j] = np.exp(-gamma * dist)

            
    # Dual optimization using gradient ascent (simplified)
    for _ in range(epochs):
        for i in range(n):
            grad = 1 - y[i] * np.sum(alpha * y * K[:, i])
            alpha[i] += lr * grad

    # Compute bias using one support vector
    support = alpha > 1e-4
    idx = np.where(support)[0][0]
    b = y[idx] - np.sum(alpha * y * K[:, idx])

    # Compute decision function for test_x
    k_test = np.zeros(n)
    for i in range(n):
        dist = np.sum((X[i] - test_x)**2)
        k_test[i] = np.exp(-gamma * dist)

    decision = np.sum(alpha * y * k_test) + b
    return 1 if decision >= 0 else -1

```
