import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        transformed_vectors = []

        for potential_matrix in x:
            # Гармонический потенциал
            is_unique_min = np.sum(potential_matrix == np.min(potential_matrix)) == 1

            # Паддинг для корректной обработки гармонического потенциала
            if is_unique_min:
                potential_matrix = np.pad(potential_matrix, 128, constant_values=20)

            # Находим границы
            indices_0, indices_1 = np.where(potential_matrix != 20)

            # Находим центр
            if is_unique_min:
                # Гармонический
                center_index = np.unravel_index(np.argmin(potential_matrix), potential_matrix.shape)
            else:
                # Квадратный и эллиптический
                center_index = (
                    (np.min(indices_0) + np.max(indices_0)) // 2,
                    (np.min(indices_1) + np.max(indices_1)) // 2
                )

            # Сдвигаем потенциал в центр
            potential_matrix = np.roll(potential_matrix, np.array(potential_matrix.shape) // 2 - center_index, axis=(0, 1))

            # Обрезаем лишнее
            if is_unique_min:
                rows, cols = potential_matrix.shape
                potential_matrix = potential_matrix[:rows//2*2, :cols//2*2].reshape(rows//2, 2, cols//2, 2).min(axis=(1, 3))

            transformed_vectors.append(potential_matrix.flatten())

        return np.array(transformed_vectors)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []

    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])

    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)

    regressor = Pipeline([
        ('transformer', PotentialTransformer()),
        ('forest', ExtraTreesRegressor(
            n_estimators=1000,
            criterion='friedman_mse',
            max_features=0.001,
            random_state=42,
            n_jobs=-1
        ))
    ])

    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)

    return {file: value for file, value in zip(test_files, predictions)}
