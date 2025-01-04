import pandas as pd
import numpy as np
from collections import Counter
import random

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree class
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _calculate_gini(self, y):
        classes = np.unique(y)
        total_samples = len(y)
        gini = 1.0
        for c in classes:
            p = len(y[y == c]) / total_samples
            gini -= p ** 2
        return gini

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        n_features_subset = int(np.sqrt(n_features))
        feature_indices = random.sample(range(n_features), n_features_subset)

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini_left = self._calculate_gini(y[left_mask])
                gini_right = self._calculate_gini(y[right_mask])
                n_left = len(y[left_mask])
                n_right = len(y[right_mask])
                n_total = len(y)
                gini_split = (n_left/n_total * gini_left + n_right/n_total * gini_right)
                gain = self._calculate_gini(y) - gini_split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if (n_samples < self.min_samples_split or depth >= self.max_depth or n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold,
                              left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

# Random Forest class
class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(pred).most_common(1)[0][0] for pred in predictions.T])
