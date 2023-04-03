from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score
import numpy as np

def information_gain_ratio(X, y, feature):
    # Calculate the information gain ratio of the given feature
    from collections import Counter
    from math import log2
    n_samples, n_features = X.shape

    # Find the unique values of the feature
    feature_values = set(X[:, feature])

    # Calculate the entropy of the whole dataset
    class_counts = Counter(y)
    class_probs = np.array(list(class_counts.values())) / len(y)
    entropy = -np.sum(class_probs * np.log2(class_probs))

    # Calculate the split entropy and the split info
    split_entropy, split_info = 0, 0
    for value in feature_values:
        # Split the dataset into two subsets based on the feature value
        left_indices = np.where(X[:, feature] < value)[0]
        right_indices = np.where(X[:, feature] >= value)[0]

        # Calculate the proportions of samples in each subset
        left_prop = len(left_indices) / n_samples
        right_prop = len(right_indices) / n_samples

        # Calculate the entropy of each subset
        left_class_counts = Counter(y[left_indices])
        left_class_probs = np.array(list(left_class_counts.values())) / len(left_indices)
        left_entropy = -np.sum(left_class_probs * np.log2(left_class_probs)) if len(left_class_probs) > 1 else 0

        right_class_counts = Counter(y[right_indices])
        right_class_probs = np.array(list(right_class_counts.values())) / len(right_indices)
        right_entropy = -np.sum(right_class_probs * np.log2(right_class_probs)) if len(right_class_probs) > 1 else 0

        # Calculate the split entropy and split info
        split_entropy += (left_prop * left_entropy + right_prop * right_entropy)
        split_info += -((left_prop * log2(left_prop)) + (right_prop * log2(right_prop)))

    # Calculate the information gain ratio
    if split_info == 0:
        return 0
    information_gain = entropy - split_entropy
    return information_gain / split_info

class C45DecisionTree(DecisionTreeClassifier):
    def __init__(self, criterion='gini', splitter='best', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                 max_features=None, random_state=None, max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, class_weight=None, 
                 information_gain_ratio=True):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         class_weight=class_weight)
        self.information_gain_ratio = information_gain_ratio

    def _get_best_split(self, X, y):
        if self.information_gain_ratio:
            criterion = 'information_gain_ratio'
        else:
            criterion = self.criterion
        return super()._get_best_split(X, y, criterion)

    def _compute_feature_importances(self):
        if self.information_gain_ratio:
            return super()._compute_feature_importances(information_gain_ratio)
        else:
            return super()._compute_feature_importances()