import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    p = np.mean(y, axis=0)
    entropy = -1 * np.sum(p * np.log(p + EPS))
    
    return entropy
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    
    return 1 - np.sum(y.mean(axis=0)**2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    
    return np.mean((y - y.mean(axis=0))**2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_left, y_left = X_subset[X_subset[:, feature_index] < threshold], y_subset[X_subset[:, feature_index] < threshold]
        X_right, y_right = X_subset[X_subset[:, feature_index] >= threshold], y_subset[X_subset[:, feature_index] >= threshold]
        
        return (X_left, y_left), (X_right, y_right)
    
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        y_left, y_right = y_subset[X_subset[:, feature_index] < threshold], y_subset[X_subset[:, feature_index] >= threshold]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        H, _ = self.all_criterions[self.criterion_name]
        
        G = 0
        feature_index = -1
        threshold = -1
        
        for i in range(X_subset.shape[1]):
            thresholds = np.unique(X_subset[:, i])[1:-1]
            for t in thresholds:
                y_left, y_right = self.make_split_only_y(i, t, X_subset, y_subset)
                G_new = H(y_subset) - y_left.shape[0] / y_subset.shape[0] * H(y_left) - y_right.shape[0] / y_subset.shape[0] * H(y_right)
                
                if G_new > G:
                    G = G_new
                    feature_index = i
                    threshold = t
        
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        ыursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if self.criterion_name == 'gini' or self.criterion_name == 'entropy':
            f = lambda y : np.argmax(np.sum(y, axis=0))
            p = lambda y : np.mean(y, axis=0)
        elif self.criterion_name == 'mad_median':
            f = lambda y: np.median(y)
            p = lambda y: np.mean(y)
        elif self.criterion_name == 'variance':
            f = lambda y: np.mean(y)
            p = lambda y: np.mean(y)
        
        self.depth += 1
        mim_leaves = X_subset.shape[0]

        if self.depth < self.max_depth and mim_leaves > self.min_samples_split:
            feature_index, threshold = self.choose_best_split(X_subset, y_subset)

            new_node = Node(feature_index, threshold)
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
            new_node.left_child = self.make_tree(X_left, y_left)
            new_node.right_child = self.make_tree(X_right, y_right)
        else:
            new_node = Node(0, 0)
            new_node.value = f(y_subset)
            new_node.proba = p(y_subset)

        self.depth -= 1
        
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def reccurent_predict(self, X, indexes, node: Node, predictions: dict, proba=False):
        if node.left_child is None:
            for i in indexes:
                if not proba:
                    predictions[i] = node.value
                else:
                    predictions[i] = node.proba
        else:
            left, right = self.make_split(node.feature_index, node.value, X, indexes)
            self.reccurent_predict(left[0], left[1], node.left_child, predictions)
            self.reccurent_predict(right[0], right[1], node.right_child, predictions)

    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        y_predicted = np.zeros(X.shape[0])

        predictions = {}
        
        tree = self.root
        indexes = np.arange(X.shape[0])
        self.reccurent_predict(X, indexes, tree, predictions)
        for idx, value in predictions.items():
            y_predicted[idx] = value
        
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))

        predictions = {}
        
        tree = self.root
        indexes = np.arange(X.shape[0])
        self.reccurent_predict(X, indexes, tree, predictions, proba=True)
        for idx, proba in predictions.items():
            y_predicted_probs[idx] = proba
        
        return y_predicted_probs
