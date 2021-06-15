from collections import Counter
from math import inf, log2
from typing import Any, Callable, List, Optional, Union

import pandas as pd


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
    return 1 - sum((count / n) ** 2 for _elem, count in c.items())


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


def information_gain(
    labels: pd.Series,
    true_idx: pd.Series,
    impurity_func: Optional[Callable] = None,
) -> float:
    """Gain in information resulting from a split of the parent node.

    Args:
        labels: The labels column of the parent node (or the dataset at the
            root).
        true_idx: A series of True, False values indicating the examples to be
            split into the left and right (True and False) children.
        impurity_func: The function used to calculate impurity. For example:
            `gini_impurity` or `entropy`.
    """

    def _calc_impurity(
        values: pd.Series,
        n_total: int,
        impurity_func: Callable,
    ) -> float:
        return len(values) / n_total * impurity_func(values)

    n_total = len(labels)
    true_impurity = _calc_impurity(labels[true_idx], n_total, impurity_func)
    false_impurity = _calc_impurity(labels[~true_idx], n_total, impurity_func)
    total_impurity = true_impurity + false_impurity
    return impurity_func(labels) - total_impurity


class DecisionTree:
    class Node:
        def __init__(
            self,
            index,
            decision_func,
            left_child,
            right_child,
        ):
            pass

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ):
        self._nodes = list()

    @property
    def nodes(self):
        return self._nodes

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    dataset = pd.DataFrame(
        {
            "carat": [0.21, 0.39, 0.5, 0.76, 0.87, 0.98, 1.13, 1.34, 1.67, 1.81],
            "price": [327, 897, 1122, 907, 2757, 2865, 3045, 3914, 4849, 5688],
            "cut": [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    print(dataset)
    true_idx = dataset["price"].apply(lambda p: p > 327 and p <= 2865)
    print(dataset[true_idx])
    # print(information_gain(dataset["cut"], true_idx, gini_impurity))
