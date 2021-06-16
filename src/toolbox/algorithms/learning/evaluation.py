from typing import Any, List, Set, Tuple, Union

import pandas as pd


def evaluate_precision(tp: int, fp: int) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0


def evaluate_precision_neg(tn: int, fn: int) -> float:
    try:
        return tn / (tn + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_sensitivity(tp: int, fn: int) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_specificity(tn: int, fp: int) -> float:
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return 0.0


def evaluate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_f1(tp: int, fp: int, fn: int) -> float:
    try:
        return 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_all(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
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
    eval_df: pd.DataFrame,
    classes: Union[List[Any], Set[Any]],
    predict_col: str,
    actual_col: str,
):
    confusion_matrix = pd.DataFrame(0, index=classes, columns=classes)
    for _ix, pred, actual in eval_df[[predict_col, actual_col]].itertuples():
        confusion_matrix.loc[actual][pred] += 1
    return confusion_matrix


def get_classification_outcomes(
    confusion_matrix: pd.DataFrame,
    classes: Set[Any],
    class_name: str,
) -> Tuple[int, int, int, int]:
    excl_idx = classes.difference(set((class_name,)))
    tp = confusion_matrix.loc[class_name, class_name]
    tn = confusion_matrix.loc[excl_idx, excl_idx].sum().sum()
    fp = confusion_matrix.loc[class_name, excl_idx].sum()
    fn = confusion_matrix.loc[excl_idx, class_name].sum()
    return (tp, tn, fp, fn)


def evaluate_classifier_per_class(
    pred_df: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    classes: Set[Any],
    actual_col: str,
) -> pd.DataFrame:
    weights = pred_df[actual_col].value_counts(normalize=True)
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
    tp = class_eval_df.loc["True Positives"].sum()
    fp = class_eval_df.loc["False Positives"].sum()
    fn = class_eval_df.loc["False Negatives"].sum()
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
        ("Micro-score", ""): evaluate_f1(tp, fp, fn),
    }
    return pd.Series(eval_dict)


def evaluate_classifier(
    pred_df: pd.DataFrame,
    predict_col: str,
    actual_col: str,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    classes = set(pred_df[predict_col].values).union(set(pred_df[actual_col].values))
    confusion_matrix = generate_confusion_matrix(
        eval_df=pred_df,
        classes=classes,
        predict_col=predict_col,
        actual_col=actual_col,
    )
    class_eval_df = evaluate_classifier_per_class(
        pred_df=pred_df,
        confusion_matrix=confusion_matrix,
        classes=classes,
        actual_col=actual_col,
    )
    model_eval_df = evaluate_classifier_model(class_eval_df)
    return (model_eval_df, class_eval_df, confusion_matrix)
