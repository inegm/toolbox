from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from math import log2
from operator import ge
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd

from toolbox.algorithms.learning.evaluation import evaluate_classifier


def gini_impurity(labels: Union[List[Any], pd.Series]) -> float:
    """Gini impurity, a measure of uncertainty used by the CART algorithm.

    Args:
        labels: The labels column of the dataset

    A measure of the probability that a label selected at random from the
    distribution of labels in a dataset would incorrectly label an example from
    that dataset. A Gini impurity of 0 indicates a perfectly certain dataset,
    that is one with a single unique label.

    The measure is calculated as 1 minus the sum of squared probabilities for
    each unique label.

    $I_{G}(p) = 1 - \sum_{i=1}^{n}p_{i}^2$

    for $\mathnormal{n}$ classes with $i \in \{1, 2, ..., n\}$
    """
    c, n = Counter(labels), len(labels)
    return 1 - sum((count / n) ** 2 for _elem, count in c.items())  # type: ignore


def entropy(labels: Union[List[Any], pd.Series]) -> float:
    """Entropy, a measure of uncertainty used by the C.45 algorithm.

    Args:
        labels: The labels column of the dataset

    A measure of the amount of information and uncertainty realized in the
    outcome of drawing a value at random. An entropy of 0 indicates a dataset
    which can yield no new information, that is one with a single unique label.

    The measure is calculated as the negation of the sum of the products of the
    probability of drawing the label and the base two log of the same probability
    for each unique label.

    $I_{G}(p) = -\sum_{i=1}^{n}p_{i} log_{2} p_{i}$

    for $\mathnormal{n}$ classes with $i \in \{1, 2, ..., n\}$
    """
    c, n = Counter(labels), len(labels)
    return -sum((count / n) * log2((count / n)) for _elem, count in c.items())


@dataclass
class DTCQuery:
    feature: str
    value: Any
    operator: Callable


@dataclass
class DTCNode:
    idx: Any
    query: Optional[DTCQuery] = None
    left: Optional[DTCNode] = None
    right: Optional[DTCNode] = None
    leaf: bool = False
    label: Optional[Any] = None


class DecisionTreeClassifier:
    """
    **Caveat**

    This is a simple, pedagogical implementation of a decision-tree classifier.
    It is not likely to win you any Kaggle competitions on its own. It
    typically achieves accuracy scores around 0.7 and is slow to train.
    That said, this is what RandomForest, Ensemble methods, and XGBoost are
    based on, so it's well worth understanding how it works.

    **Fitting the tree**

    These classifiers work by recursively splitting a dataset using **queries**
    on the features in such a way as to maximize **information gain** at each new
    split. Information gain is calculated for each split by using an **impurity
    function**, which is sometimes known as an *error function*. Queries can be
    as complex as the implementer cares to make them, but in this implementation
    they are simple : all queries here are of the form $v \ge w$. A value $w$ is
    selected from the features and a candidate split is created : all examples
    for which the query yields True go to the left child and all others go to the
    right child. The candidate information gain is then evaluated. This process
    is repeated for each feature and for each unique feature value. The actual
    split is chosen to be the candidate with the highest information gain. For
    more on information gain, see the [information_gain]
    [toolbox.algorithms.learning.decisiontree.DecisionTreeClassifier.information_gain]
    method's documentation. The leaves of the tree contain subsets of the dataset
    with no impurity, that is to say all examples of the subset belong to a
    single class.

    **Making predictions**

    By traveling through the tree with a novel, unlabeled example, applying the
    queries to its features' values to determine the path, we end up at a leaf
    node which determines the predicted class label for the novel example.

    Examples:

        >>> import pandas as pd
        >>> url = "/".join([
            "https://archive.ics.uci.edu",
            "ml/machine-learning-databases",
            "heart-disease/processed.cleveland.data"])
        >>> df = pd.read_csv(
                url,
                header=None,
                names=[
                    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
                ]
            )
        >>> df['target'].replace({2: 1, 3: 1, 4:1}, inplace=True)
        >>> train = df[:201]
        >>> test = df[201:]
        >>> dtc = DecisionTreeClassifier(train, 'target', gini_impurity)
        >>> dtc.fit()  # 2 minute coffee break
        >>> _ = dtc.evaluate(test)

            MODEL

            Precision           mean        0.688657
                                weighted    0.696078
            Precision negative  mean        0.688657
                                weighted    0.681236
            Sensitivity         mean        0.703326
                                weighted    0.701773
            Specificity         mean        0.703326
                                weighted    0.704880
            Accuracy            mean        0.696078
                                weighted    0.696078
            F1-score            mean        0.687395
                                weighted    0.690460
            dtype: float64

            CLASSES

                                        0          1
            True Positives      44.000000  27.000000
            True Negatives      27.000000  44.000000
            False Positives     10.000000  21.000000
            False Negatives     21.000000  10.000000
            Precision            0.814815   0.562500
            Precision negative   0.562500   0.814815
            Sensitivity          0.676923   0.729730
            Specificity          0.729730   0.676923
            Accuracy             0.696078   0.696078
            F1-score             0.739496   0.635294
            Weight (actual)      0.529412   0.470588

            CONFUSION MATRIX

                    0   1
            actual
            0       44  10
            1       21  27

    **A comparison**

    The accuracy score for the given example is of 0.70, which is not far from
    scikit-learn's DecisionTreeClassifier accuracy of 0.76 given the same dataset.
    That said, with hyperparameter tuning (not possible here), the scikit-learn
    implementation can reach much higher accuracy scores (above 0.85).
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        labels_col: str,
        impurity_func: Callable,
    ):
        """
        Args:
            dataset: Including one or more feature columns and a single labels
                column.
            labels_col: The name of the labels column.
            impurity_func: Options include [gini_impurity]
                [toolbox.algorithms.learning.decisiontree.gini_impurity] and
                [entropy][toolbox.algorithms.learning.decisiontree.entropy],
                but any callable with the same signature which returns a float
                will do.
        """
        self.features = dataset.drop(labels_col, axis=1)
        self.labels = dataset[labels_col]
        self.impurity_func = impurity_func
        self._tree = None

    @property
    def tree(self):
        """Pointer to the decision-tree's root node.

        The (binary) tree can be traversed by traveling through each node's
        left (True) and right (False) child nodes.
        """
        return self._tree

    def information_gain(self, idx: Any, true_idx: Any) -> float:
        """Gain in information resulting from a split of the parent node.

        Weighted impurity is calculated for each side of the split, using the
        impurity function set at the initialization of the model, and the total
        impurity of the split is determined by the sum of these. The information
        gain is the difference of the impurity of the parent and this sum :

        $gain = I_{p} - w_{l}I_{l} + w_{r}I_{r}$

        The weights are calculated as the fraction of unique labels present in
        the split relative to the model's set of all known labels.

        Args:
            idx (pd.array): The (pandas array) index of the examples of the node
                before the split.
            true_idx (pd.array): The (pandas array) index of the node's examples
                for which the query condition is true. ie.: The left child's
                index after the split.
        """

        def _calc_impurity(
            values: pd.Series,
            n_total: int,
            impurity_func: Callable,
        ) -> float:
            return len(values) / n_total * impurity_func(values)

        labels = self.labels.loc[idx]
        n_total = len(labels)
        true_impurity = _calc_impurity(labels[true_idx], n_total, self.impurity_func)
        false_impurity = _calc_impurity(labels[~true_idx], n_total, self.impurity_func)
        total_impurity = true_impurity + false_impurity
        return self.impurity_func(labels) - total_impurity

    def find_best_split(self, idx: Any):
        """Find the split which yields the highest information gain.

        Args:
            idx (pd.array): The (pandas array) index of the examples of the node
                before the split.
        """
        max_gain = 0.0
        query = None
        true_idx = None
        false_idx = None
        for feature in self.features.columns:
            for value in self.features[feature].unique():
                split_condition = self.features.loc[idx][feature].ge(value)
                n_true = split_condition.value_counts().get(True, 0)
                n_false = split_condition.value_counts().get(False, 0)
                if (n_true == 0) or (n_false == 0):
                    continue
                gain = self.information_gain(idx=idx, true_idx=split_condition)
                if gain >= max_gain:
                    true_idx = self.features.loc[idx][split_condition].index
                    false_idx = self.features.loc[idx][~split_condition].index
                    max_gain = gain
                    query = DTCQuery(feature, value, ge)
        return (max_gain, query, true_idx, false_idx)

    def fit(self):
        """Build the decision tree."""

        def _fit(node):
            gain, query, left_idx, right_idx = self.find_best_split(node.idx)
            if gain == 0:
                return DTCNode(
                    node.idx, leaf=True, label=self.labels.loc[node.idx].values[0]
                )
            node.query = query
            node.left = _fit(DTCNode(left_idx))
            node.right = _fit(DTCNode(right_idx))
            return node

        self._tree = _fit(DTCNode(self.features.index))

    def evaluate(
        self,
        data: pd.DataFrame,
        verbose=True,
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """Evaluate the performance of the decision-tree model.

        See
        [evaluation.evaluate_classifier]
        [toolbox.algorithms.learning.evaluation.evaluate_classifier].

        Args:
            data: A labeled DataFrame. The label column is expected to be named
                the same as the labels Series that was provided at initialization.
        """
        predict_col = str(self.labels.name)
        actual_col = "_".join([predict_col, "actual"])

        pred_df = self.predict(data.drop(predict_col, axis=1))
        pred_df[actual_col] = data[predict_col]

        model_eval, class_eval, confusion_matrix = evaluate_classifier(
            predict_df=pred_df,
            predict_col=predict_col,
            actual_col=actual_col,
            verbose=verbose,
        )
        return (model_eval, class_eval, confusion_matrix)

    def predict(self, data: pd.DataFrame):
        """Predict classes for a set of unlabeled examples.

        Args:
            data: Unlabeled DataFrame of examples.
        """
        labels = list()
        for _ix, row in data.iterrows():
            node = self.tree
            while not node.leaf:
                if node.query.operator(row[node.query.feature], node.query.value):
                    node = node.left
                else:
                    node = node.right
            labels.append(node.label)
        data[self.labels.name] = labels
        return data
