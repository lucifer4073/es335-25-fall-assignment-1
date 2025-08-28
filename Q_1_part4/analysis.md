# Decision Tree Training & Prediction Complexity – Trend Analysis

We analyze the experimental plots of training and prediction times for different input/output types in decision trees:

- **X (features):** Discrete vs Real
- **y (labels):** Discrete (classification) vs Real (regression)

## Decision Tree Training and Prediction Complexity

The **training time complexity** of a decision tree depends mainly on the number of samples \(N\) and the number of features \(M\).  

- For **discrete (categorical) features**, each split requires scanning through the categories, which costs about \(O(N)\) per feature, giving a total training cost of roughly \(O(NM)\) for a balanced tree.  
- For **real-valued features**, the algorithm must evaluate many possible thresholds per feature (up to \(O(N)\) thresholds), which increases the cost to about \(O(MN^2)\). This makes regression or classification with continuous features significantly more expensive.  

In contrast, the **prediction time complexity** is much lighter. Once a tree is built, classifying or predicting a single sample only requires traversing from the root to a leaf, which takes time proportional to the depth of the tree. For a balanced decision tree, the depth is about \(O(\log N)\). Thus, predicting \(N\) samples costs about \(O(N \log N)\), and is largely independent of the number of features \(M\).

## Trends

- **Discrete features:** Training grows ~linearly in both \(N\) and \(M\).  
- **Real features:** Training grows ~quadratically in \(N\), linearly in \(M\), making it much more expensive.  
- **Prediction:** Always cheap — ~linear in \(N\), almost flat in \(M\). Depth governs cost, and depth grows slowly (≈ \(O(\log N)\), weak dependence on M).

Thus the experiments align well with the theoretical expectations:

| Case                  | Training Complexity | Prediction Complexity |
|-----------------------|---------------------|------------------------|
| Discrete X, Discrete y | \(O(NM)\)          | \(O(N \log N)\)        |
| Discrete X, Real y     | \(O(NM)\)          | \(O(N \log N)\)        |
| Real X, Discrete y     | \(O(MN^2)\)        | \(O(N \log N)\)        |
| Real X, Real y         | \(O(MN^2)\)        | \(O(N \log N)\)        |