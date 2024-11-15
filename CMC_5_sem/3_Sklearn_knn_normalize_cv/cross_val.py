import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    indices = np.arange(num_objects)
    fold_size = num_objects // num_folds
    folds = []

    for i in range(num_folds):
        start = i * fold_size
        if i < num_folds - 1:
            end = start + fold_size
        else:
            end = num_objects
        val_indices = indices[start:end]
        train_indices = np.array([index for index in indices if index < start or index >= end])
        folds.append((train_indices, val_indices))

    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    scores = defaultdict(list)

    """
    for each combination of hyperparameters:
    1.1) the data is divided into training and validation samples
    1.2) if a normalizer is selected, the data is normalized
    2) the model is trained on training data and makes predictions on validation data
    3.1) the predicted labels are compared with the true ones and a score is calculated
    3.2) scores for all folds are collected
    4) for each combination of hyperparameters, the average score across all folds is calculated
    """

    for n_neighbors, metric, weight, (normalizer, normalizer_name) in (
            (n_neighbors, metric, weight, (normalizer, normalizer_name))
            for n_neighbors in parameters['n_neighbors']
            for metric in parameters['metrics']
            for weight in parameters['weights']
            for normalizer, normalizer_name in parameters['normalizers']):

        fold_scores = []

        for train_indices, val_indices in folds:
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            
            if normalizer:
                normalizer.fit(X_train)
                X_train = normalizer.transform(X_train)
                X_val = normalizer.transform(X_val)

            knn_model = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
            knn_model.fit(X_train, y_train)
            y_pred = knn_model.predict(X_val)

            fold_score = score_function(y[val_indices], y_pred)
            fold_scores.append(fold_score)

        mean_score = np.mean(fold_scores)
        key = (normalizer_name, n_neighbors, metric, weight)
        scores[key].append(mean_score)

        mean_scores = {}

        for key, score in scores.items():
            mean_scores[key] = np.mean(score)

    return mean_scores