from typing import Any, List, Set, Tuple, Union

import pandas as pd


def evaluate_precision(tp: int, fp: int) -> float:
    """Precision, aka Positive Predictive Value (PPV).


    $PPV=\dfrac{TP}{TP + FP}$

    Args:
        tp: True Positives
        fp: False Positives
    """
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0


def evaluate_precision_neg(tn: int, fn: int) -> float:
    """Negative precision, aka Negative Predictive Value (NPV).


    $NPV=\dfrac{TN}{TN + FN}$

    Args:
        tn: True Negatives
        fn: False Negatives
    """
    try:
        return tn / (tn + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_sensitivity(tp: int, fn: int) -> float:
    """Sensitivity, aka Recall, aka True Positive Rate (TPR).

    $TPR=\dfrac{TP}{TP + FN}$

    Args:
        tp: True Positives
        fn: False Negatives
    """
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_specificity(tn: int, fp: int) -> float:
    """Specificity, aka True Negative Rate (TNR).

    $TNR=\dfrac{TP}{TP + FP}$

    Args:
        tp: True Positives
        fp: False Positives
    """
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return 0.0


def evaluate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Accuracy (ACC).

    $ACC=\dfrac{TP + TN}{TP + TN + FP + FN}$

    Args:
        tp: True Positives
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
    """
    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_f1(tp: int, fp: int, fn: int) -> float:
    """F1-score.

    *F1-score* $=\dfrac{2TP}{2TP + FP + FN}$

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
    """
    try:
        return 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_all(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """
    Evaluates the following statistical measures:

    - [precision]
    [toolbox.algorithms.learning.evaluation.evaluate_precision]
    - [negative precision]
    [toolbox.algorithms.learning.evaluation.evaluate_precision_neg]
    - [sensitivity (recall)]
    [toolbox.algorithms.learning.evaluation.evaluate_sensitivity]
    - [specificity]
    [toolbox.algorithms.learning.evaluation.evaluate_specificity]
    - [accuracy]
    [toolbox.algorithms.learning.evaluation.evaluate_accuracy]
    - [f1-score]
    [toolbox.algorithms.learning.evaluation.evaluate_f1]

    Args:
        tp: True Positives
        tn: True Negatives
        fp: False Positives
        fn: False Negatives

    Returns:
        See source for dictionary keys.
    """
    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": evaluate_precision(tp, fp),
        "Precision negative": evaluate_precision_neg(tn, fn),
        "Sensitivity": evaluate_sensitivity(tp, fn),
        "Specificity": evaluate_specificity(tn, fp),
        "Accuracy": evaluate_accuracy(tp, tn, fp, fn),
        "F1-score": evaluate_f1(tp, fp, fn),
    }


def generate_confusion_matrix(
    predict_df: pd.DataFrame,
    classes: Set[Any],
    predict_col: str,
    actual_col: str,
) -> pd.DataFrame:
    """
    The confusion matrix summarises how well a model labels examples belonging
    to a set of classes. In the case of binary classification - where there are
    two classes - the confusion matrix is a $2*2$ matrix. More generally: for
    $n$ classes the matrix will have a shape of $n*n$. Here, the *predicted*
    labels determine the column headers and the *actual* labels determine the row
    headers.

    The matrix is used to calculate the classification outcomes. See
    [get_classification_outcomes]
    [toolbox.algorithms.learning.evaluation.get_classification_outcomes]

    Args:
        predict_df: A DataFrame containing a set of features, their classes as
            predicted by the classification model, and the actual classes
            correctly labeling each example.
        classes: The set of unique class labels, being the union of the unique
            actual and predicted class labels.
        predict_col: The name of the column holding the predicted classes.
        actual_col: The name of the column holding the actual classes.

    Returns:
        For $n$ classes, an $n*n$ pd.DataFrame with *predicted* labels determining
        the column headers and the *actual* labels determining the row headers.
    headers

    Examples:

        Given a simple binary classifier with two classes 'Yes' and 'No', the
        following is a possible result:
        ```
                No  Yes
        actual
        No      122   49
        Yes      49   74
        ```
    """
    confusion_matrix = pd.DataFrame(0, index=classes, columns=classes)
    for _ix, pred, actual in predict_df[[predict_col, actual_col]].itertuples():
        confusion_matrix.loc[actual][pred] += 1
    confusion_matrix.index.name = "actual"
    return confusion_matrix


def get_classification_outcomes(
    confusion_matrix: pd.DataFrame,
    classes: Set[Any],
    class_name: str,
) -> Tuple[int, int, int, int]:
    """
    Given a confusion matrix, this function counts the cases of:

    - **True Positives** : classifications that accurately labeled a class
    - **True Negatives** : classifications that accurately labeled an example as
        not belonging to a class.
    - **False Positives** : classifications that attributed the wrong label to an
        example.
    - **False Negatives** : classifications that falsely claimed that an example
        does not belong to a class.

    Args:
        confusion_matrix: The result of calling [generate_confusion_matrix]
            [toolbox.algorithms.learning.evaluation.generate_confusion_matrix]
        classes: The set of all class labels
        class_name: The name (label) of the class being evaluated.

    Returns:
        - `tp`: Count of True Positives
        - `tn`: Count of True Negatives
        - `fp`: Count of False Positives
        - `fn`: Count of False Negatives
    """
    excl_idx = classes.difference(set((class_name,)))
    tp = confusion_matrix.loc[class_name, class_name]
    tn = confusion_matrix.loc[excl_idx, excl_idx].sum().sum()
    fp = confusion_matrix.loc[class_name, excl_idx].sum()
    fn = confusion_matrix.loc[excl_idx, class_name].sum()
    return (tp, tn, fp, fn)


def evaluate_classifier_per_class(
    predict_df: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    classes: Set[Any],
    actual_col: str,
) -> pd.DataFrame:
    """
    Evaluates all (see [evaluate_all]
    [toolbox.algorithms.learning.evaluation.evaluate_all]) statistical measures,
    the classification outcomes (see [get_classification_outcomes]
    [toolbox.algorithms.learning.evaluation.get_classification_outcomes]), and
    the relative weights for each class label.

    Args:
        predict_df: A DataFrame containing a set of features, their classes as
            predicted by the classification model, and the actual classes
            correctly labeling each example.
        confusion_matrix: The result of calling [generate_confusion_matrix]
            [toolbox.algorithms.learning.evaluation.generate_confusion_matrix]
        classes: The set of all class labels
        actual_col: The name of the column holding the actual classes.

    Returns:
        With the statistical measure names determining the index and a column
        per class label.

    Examples:
        ```
                                    No         Yes
        True Positives      122.000000   74.000000
        True Negatives       74.000000  122.000000
        False Positives      49.000000   49.000000
        False Negatives      49.000000   49.000000
        Precision             0.713450    0.601626
        Precision negative    0.601626    0.713450
        Sensitivity           0.713450    0.601626
        Specificity           0.601626    0.713450
        Accuracy              0.666667    0.666667
        F1-score              0.713450    0.601626
        Weight (actual)       0.581633    0.418367
        ```
    """
    weights = predict_df[actual_col].value_counts(normalize=True)
    class_eval: dict = {}
    for class_name in classes:
        class_eval.update({class_name: {}})
        tp, tn, fp, fn = get_classification_outcomes(
            confusion_matrix=confusion_matrix,
            classes=classes,
            class_name=class_name,
        )
        eval_dict = evaluate_all(tp, tn, fp, fn)
        eval_dict.update({"Weight (actual)": weights.get(class_name, 0)})
        class_eval[class_name].update(eval_dict)
    return pd.DataFrame(class_eval).fillna(0)


def evaluate_classifier_model(class_eval_df: pd.DataFrame) -> pd.Series:
    """
    Evaluates all (see [evaluate_all]
    [toolbox.algorithms.learning.evaluation.evaluate_all]) statistical measures,
    both *mean* across and *weighted* by class.

    Args:
        class_eval_df: Classifier evaluation per class, as generated by
            [evaluate_classifier_per_class]
            [toolbox.algorithms.learning.evaluation.evaluate_classifier_per_class]

    Returns:
        With a multi-index, of which the 0 level is the statistical measure and
        the 1 level is the method.

    Examples:
        ```
        Precision           mean        0.657538
                            weighted    0.666667
        Precision negative  mean        0.657538
                            weighted    0.648410
        Sensitivity         mean        0.657538
                            weighted    0.666667
        Specificity         mean        0.657538
                            weighted    0.648410
        Accuracy            mean        0.666667
                            weighted    0.666667
        F1-score            mean        0.657538
                            weighted    0.666667
        ```
    """
    weights = class_eval_df.loc["Weight (actual)"]
    eval_dict = {
        ("Precision", "mean"): class_eval_df.loc["Precision"].mean(),
        ("Precision", "weighted"): (class_eval_df.loc["Precision"] * weights).sum(),
        ("Precision negative", "mean"): class_eval_df.loc["Precision negative"].mean(),
        ("Precision negative", "weighted"): (
            class_eval_df.loc["Precision negative"] * weights
        ).sum(),
        ("Sensitivity", "mean"): class_eval_df.loc["Sensitivity"].mean(),
        ("Sensitivity", "weighted"): (class_eval_df.loc["Sensitivity"] * weights).sum(),
        ("Specificity", "mean"): class_eval_df.loc["Specificity"].mean(),
        ("Specificity", "weighted"): (class_eval_df.loc["Specificity"] * weights).sum(),
        ("Accuracy", "mean"): class_eval_df.loc["Accuracy"].mean(),
        ("Accuracy", "weighted"): (class_eval_df.loc["Accuracy"] * weights).sum(),
        ("F1-score", "mean"): class_eval_df.loc["F1-score"].mean(),
        ("F1-score", "weighted"): (class_eval_df.loc["F1-score"] * weights).sum(),
    }
    return pd.Series(eval_dict)


def evaluate_classifier(
    predict_df: pd.DataFrame,
    predict_col: str,
    actual_col: str,
    verbose: bool = True,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a classification model overall and for each of the its classes. It
    is capable of evaluating both binary and multi-class classifiers.

    This function wraps:

    - [evaluate_classifier_model]
    [toolbox.algorithms.learning.evaluation.evaluate_classifier_model]
    - [evaluate_classifier_per_class]
    [toolbox.algorithms.learning.evaluation.evaluate_classifier_per_class]
    - [generate_confusion_matrix]
    [toolbox.algorithms.learning.evaluation.generate_confusion_matrix]

    Args:
        predict_df: A DataFrame containing a set of features, their classes as
            predicted by the classification model, and the actual classes
            correctly labeling each example.
        predict_col: The name of the column holding the predicted classes.
        actual_col: The name of the column holding the actual classes.
        verbose: Whether or not to print out the results.

    Returns:
        A tuple containing:

        - `model_eval`
        - `class_eval`
        - `confusion_matrix`

    Examples:

        ```
        MODEL

        Precision           mean        0.657538
                            weighted    0.666667
        Precision negative  mean        0.657538
                            weighted    0.648410
        Sensitivity         mean        0.657538
                            weighted    0.666667
        Specificity         mean        0.657538
                            weighted    0.648410
        Accuracy            mean        0.666667
                            weighted    0.666667
        F1-score            mean        0.657538
                            weighted    0.666667

        CLASSES

                                    No         Yes
        True Positives      122.000000   74.000000
        True Negatives       74.000000  122.000000
        False Positives      49.000000   49.000000
        False Negatives      49.000000   49.000000
        Precision             0.713450    0.601626
        Precision negative    0.601626    0.713450
        Sensitivity           0.713450    0.601626
        Specificity           0.601626    0.713450
        Accuracy              0.666667    0.666667
        F1-score              0.713450    0.601626
        Weight (actual)       0.581633    0.418367

        CONFUSION MATRIX

                No  Yes
        actual
        No      122   49
        Yes      49   74
        ```
    """
    classes = set(predict_df[predict_col].values).union(
        set(predict_df[actual_col].values)
    )
    confusion_matrix = generate_confusion_matrix(
        predict_df=predict_df,
        classes=classes,
        predict_col=predict_col,
        actual_col=actual_col,
    )
    class_eval = evaluate_classifier_per_class(
        predict_df=predict_df,
        confusion_matrix=confusion_matrix,
        classes=classes,
        actual_col=actual_col,
    )
    model_eval = evaluate_classifier_model(class_eval)
    if verbose:
        print("\nMODEL\n")
        print(model_eval)
        print("\nCLASSES\n")
        print(class_eval)
        print("\nCONFUSION MATRIX\n")
        print(confusion_matrix, "\n")
    return (model_eval, class_eval, confusion_matrix)
