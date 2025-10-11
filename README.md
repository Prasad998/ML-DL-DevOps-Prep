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
                                                                                # norm does it back to (n,m) dimension after the operation. 

        # Assign each point to the nearest centroid       
        labels = np.argmin(distances, axis=1)
        # Update centroids
        for i in range(k):                                                      #Loop over each cluster.
            if np.any(labels == i):                                             #Check if there are any points assigned to this cluster.
                centroids[i] = X[labels == i].mean(axis=0)                      #Selects all points assigned to cluster i & Computes the new centroid as the mean of all points in cluster i.
                
    # Step 3: Count how many points fall into each cluster
    cluster_sizes = np.bincount(labels, minlength=k)
    return sorted(cluster_sizes.tolist())
```

---

[4. PCA Implementation](https://products.123ofai.com/qnalab/problems/pca-implementation)


**One-line summary:** PCA finds the top-k directions (principal components) that maximize data variance by eigen-decomposing the covariance matrix, then projects mean-centered data onto those directions.

```python
import numpy as np

def PCA(X, k):
    
    X = np.asarray(X, dtype=float)                     # Step 1: Convert input data to a NumPy float array
    X_centered = X - np.mean(X, axis=0)                # Step 2: Center the data by subtracting the mean of each feature
    cov = np.cov(X_centered, rowvar=False)             # Step 3: Compute the covariance matrix of centered data
    eigenvalues, eigenvectors = np.linalg.eigh(cov)    # Step 4: Find eigenvalues and eigenvectors of the covariance matrix
    sorted_indices = np.argsort(eigenvalues)[::-1]     # Step 5: Sort eigenvalues (and corresponding eigenvectors) in descending order
    top_k = eigenvectors[:, sorted_indices[:k]]        # Step 6: Select the top-k eigenvectors (principal components)
    reduced = X_centered @ top_k                       # Step 7: Project the centered data onto the top-k principal components
    return reduced.shape                               # Step 8: Return the shape of reduced data (n_samples, k_features)
```

### 🧠 Summary of what happens mathematically:

1. **Mean-center** → $$( X_c = X - \mu )$$
2. **Covariance** → $$( \Sigma = \frac{1}{n-1} X_c^T X_c )$$
3. **Eigen-decomposition** → $$( \Sigma v_j = \lambda_j v_j )$$
4. **Select top-k eigenvectors** $$(largest (\lambda_j))$$
5. **Project** → $$( Y = X_c V_k )$$


Result: $$( Y \in \mathbb{R}^{n \times k} )$$ → lower-dimensional representation capturing maximum variance.

---
| Step | Equation                                                       | Meaning                                            |
| ---- | -------------------------------------------------------------- | -------------------------------------------------- |
| 1    | $\mu = \frac{1}{n}\sum_i x_i$                            | overall mean                                       |
| 2    | $( \mu_c = \frac{1}{n_c}\sum_{x_i \in c} x_i )$                  | mean per class                                     |
| 3    | $( S_w = \sum_c \sum_{x_i \in c} (x_i - \mu_c)(x_i - \mu_c)^T )$ | within-class scatter                               |
| 4    | $( S_b = \sum_c n_c(\mu_c - \mu)(\mu_c - \mu)^T ) $              | between-class scatter                              |
| 5    | Solve $( S_w^{-1}S_b w = \lambda w )  $                          | generalized eigenvalue problem                     |
| 6    | Choose $( w_{max} )$ for largest eigenvalue                      | best direction for discrimination                  |
| 7    | Project $( Y = X w_{max} ) $                                      | new 1-D representation maximizing class separation |

[5. LDA Implementation using NumPy](https://products.123ofai.com/qnalab/problems/lda-implementation-using-numpy)

```py
import numpy as np

def lda(X, y):
    X = np.asarray(X, dtype=float)                     # Convert input data to float array → shape (n_samples, n_features)
    y = np.asarray(y)                                  # Convert labels to NumPy array

    class_labels = np.unique(y)                        # Get all unique class labels
    mean_overall = np.mean(X, axis=0)                  # Compute overall mean vector μ

    S_w = np.zeros((X.shape[1], X.shape[1]))           # Initialize within-class scatter matrix (S_w)
    S_b = np.zeros((X.shape[1], X.shape[1]))           # Initialize between-class scatter matrix (S_b)

    for c in class_labels:                             # Loop over each class
        X_c = X[y == c]                                # Select samples belonging to class c
        mean_c = np.mean(X_c, axis=0)                  # Compute class mean vector μ_c
        S_w += (X_c - mean_c).T @ (X_c - mean_c)       # Update within-class scatter: S_w += Σ(x - μ_c)(x - μ_c)ᵀ

        n_c = X_c.shape[0]                                 # Number of samples in class c
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_b += n_c * (mean_diff @ mean_diff.T)             # Update between-class scatter: S_b += n_c(μ_c - μ)(μ_c - μ)ᵀ

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)   # Solve eigenproblem for S_w⁻¹S_b → eig_vecs are discriminant directions
    top_vec = eig_vecs[:, np.argmax(eig_vals)].reshape(-1, 1)       # Select eigenvector with largest eigenvalue (maximizes class separability)

    projected = X @ top_vec                          # Project data onto top discriminant direction: Y = Xw
    return projected.shape                            # Return the shape of reduced data (n_samples, 1)
```
