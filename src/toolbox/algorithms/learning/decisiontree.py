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
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        impurity_func: Callable,
    ):
        self.features = features
        self.labels = labels
        self.impurity_func = impurity_func
        self._tree = None

    @property
    def tree(self):
        """The decision-tree's root node."""
        return self._tree

    def information_gain(self, idx: Any, true_idx: Any) -> float:
        """Gain in information resulting from a split of the parent node.

        Args:
            idx (pd.array): The (pandas array) index of the examples of the node
                before the split.
            true_idx (pd.array): The (pandas array) index of the node's examples
                for which the query condition is true. ie.: The left child after
                the split.
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
        max_gain = 0
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
        """Evaluate the decision-tree model.

        See
        [evaluation.evaluate_classifier()]
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
