from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    return [row[120:500:5] for row in X[::4]]


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    total = 0
    flag = False
    
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            total += X[i][i]
            flag = True
            
    return total if flag else -1


def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    X_copy = deepcopy(X)
    n, m = len(X), len(X[0])

    for j in range(m):
        mean = sum(X[i][j] for i in range(n)) / n
        
        for i in range(n):
            if (X_copy[i][j] < 0.25 * mean) or (X_copy[i][j] > 1.5 * mean):
                X_copy[i][j] = -1

    return X_copy