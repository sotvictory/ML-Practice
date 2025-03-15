import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype
        self.is_fitted = False

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """

        X = X.fillna('missing')
        self.unique_values = [sorted(X[column].unique()) for column in X.columns]
        self.is_fitted = True

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """

        X = X.fillna('missing')

        if not self.is_fitted:
            raise ValueError("You must fit the encoder before transforming data.")

        encoded_columns = []

        for column_index, column in enumerate(X.columns):
            unique_values = self.unique_values[column_index]
            one_hot_encoded = np.zeros((X.shape[0], len(unique_values)), dtype=self.dtype)

            value_to_index = {value: idx for idx, value in enumerate(unique_values)}
            indices = X[column].map(value_to_index).fillna(-1).astype(int)

            one_hot_encoded[np.arange(X.shape[0]), indices] = 1.0

            encoded_columns.append(one_hot_encoded)

        return np.hstack(encoded_columns)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.is_fitted = False

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """

        X = X.fillna('missing')

        self.n_objects, self.n_features = X.shape
        self.success_means = [{} for _ in range(self.n_features)]
        self.value_frequencies = [{} for _ in range(self.n_features)]

        for feature_index in range(self.n_features):
            for feature_value in X.iloc[:, feature_index].unique():
                filtered_target_values = Y[X.iloc[:, feature_index] == feature_value]
                self.success_means[feature_index][feature_value] = filtered_target_values.mean()
                self.value_frequencies[feature_index][feature_value] = (X.iloc[:, feature_index] == feature_value).mean()

        self.is_fitted = True

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """

        X = X.fillna('missing')

        if not self.is_fitted:
            raise ValueError("You must fit the encoder before transforming data.")

        n_objects, n_features = X.shape
        encoded_columns = np.zeros((n_objects, 3 * n_features), dtype=self.dtype)

        for feature_index in range(n_features):
            feature_column = X.iloc[:, feature_index]

            success_counts = feature_column.map(self.success_means[feature_index]).fillna(0)
            value_frequencies = feature_column.map(self.value_frequencies[feature_index]).fillna(0)

            relation = (success_counts + a) / (value_frequencies + b)
            relation[value_frequencies == 0] = 0

            encoded_columns[:, 3 * feature_index] = success_counts
            encoded_columns[:, 3 * feature_index + 1] = value_frequencies
            encoded_columns[:, 3 * feature_index + 2] = relation

        return encoded_columns

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds = [(train_indices, val_indices) for train_indices, val_indices in group_k_fold(X.shape[0], n_splits=self.n_folds, seed=seed)]
        self.transformers = [SimpleCounterEncoder(dtype=self.dtype) for _ in range(self.n_folds)]

        for fold_index in range(self.n_folds):
            train_indices = self.folds[fold_index][1]
            self.transformers[fold_index].fit(X.iloc[train_indices, :], Y.iloc[train_indices])

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        transformed_data = np.zeros((X.shape[0], X.shape[1] * 3))

        for fold_index in range(self.n_folds):
            train_indices = self.folds[fold_index][0]
            transformed_data[train_indices, :] = self.transformers[fold_index].transform(X.iloc[train_indices, :], a, b)

        return transformed_data

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    unique_values = np.unique(x)
    optimal_weights = np.zeros_like(unique_values, dtype='f')

    for i in range(len(unique_values)):
        optimal_weights[i] = (y[x == unique_values[i]].sum() / (x == unique_values[i]).sum())

    return optimal_weights