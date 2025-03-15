import numpy as np


def get_part_of_array(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    return X[::4, 120:500:5]


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag = np.diagonal(X)
    non_neg_diag = diag[diag >= 0]
    return np.sum(non_neg_diag) if non_neg_diag.size > 0 else -1


def replace_values(X: np.ndarray) -> np.ndarray:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    X_copy = np.copy(X)
    n, m = X.shape
    
    for j in range(m):
        mean = np.mean(X[:, j])
        X_copy[(X[:, j] < 0.25 * mean) | (X[:, j] > 1.5 * mean), j] = -1
    
    return X_copy