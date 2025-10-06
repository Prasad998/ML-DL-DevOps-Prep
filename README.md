My Resources for the preparation of Machine Learning and Deep Learning Interviews
[Shaw Talebi](https://github.com/ShawhinT)  

# [123ofAI -  ML coding Problems](https://products.123ofai.com/qnalab/problems)


### Supervised Learning
---
[1. Kernel SVM using numpy](https://products.123ofai.com/qnalab/problems/kernel-svm-using-numpy)


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
[2. Linear Regression](https://products.123ofai.com/qnalab/problems/linear-regression)

Implement Linear Regression using the closed-form solution (Normal Equation) with NumPy.  
You are given:
* A **feature matrix** `X` of shape `(n, d)`, where
  * `n` = number of training samples
  * `d` = number of features
* A **target vector** `y` of shape `(n,)`
Your task:
1. **Add a bias column** (a column of ones) to `X`.
2. Compute the optimal weight vector `w` using the **Normal Equation**:
3. Return the learned weight vector `w` (including the bias term), rounded to **2 decimal places**.
```python
Input:
X = [[1], [2], [3]]
y = [2, 4, 6]
print(linear_regression(X, y))

Output:
[0.0, 2.0]
```
```py
import numpy as np

def linear_regression_fit(X, y):
    X = np.asarray(X, dtype=float)      # Convert features to NumPy float array
    y = np.asarray(y, dtype=float)      # Convert target to NumPy float array

    n = X.shape[0]                      # Number of samples (rows)
    X_bias = np.hstack((np.ones((n, 1)), X))  # Add bias (column of ones) to X

    # Compute weights using Normal Equation: w = (X^T X)^+ X^T y
    w = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    w = np.round(w, 2)                  # Round weights to 2 decimal places
    w[w == -0.0] = 0.0                  # Replace -0.0 with 0.0 for clean output
    return w.tolist()                   # Return weights as Python list

```
---
### Unsupervised Learning
---
[3. K-Means Clustering](https://products.123ofai.com/qnalab/problems/k-means-clustering)
```
Implement the K-Means clustering algorithm from scratch using NumPy.
You are given:
A dataset X with n samples and m features
An integer k, representing the number of clusters

Your task is to:
1. Run the K-Means algorithm for 100 iterations (no convergence check)
2. Assign each sample to the nearest cluster
3. Count how many samples fall into each cluster and return a sorted list of cluster sizes
```
```py
import numpy as np

def k_means_cluster_sizes(X, k):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    np.random.seed(0)  # ensure reproducibility

    # Step 1: Initialize centroids randomly from the dataset
    centroids = X[np.random.choice(n, k, replace=False)]                        #Purpose: randomly select k unique indices from 0 to n-1.

    
    # Step 2: Run K-Means for 100 iterations
    for _ in range(100):
        # Compute distances of each point to each centroid
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)              #Adds a new axis to X #If X has shape (n, m) → X[:, None] has shape (n, 1, m).

        # Assign each point to the nearest centroid       
        labels = np.argmin(distances, axis=1)
        # Update centroids
        for i in range(k):                                                      #Loop over each cluster.
            if np.any(labels == i):                                             #Check if there are any points assigned to this cluster.
                centroids[i] = X[labels == i].mean(axis=0)                      #Selects all points assigned to cluster i & Computes the new centroid as the mean of all points in cluster i.
                
    # Step 3: Count how many points fall into each cluster
    counts = np.bincount(labels, minlength=k)
    return sorted(counts.tolist())
```
