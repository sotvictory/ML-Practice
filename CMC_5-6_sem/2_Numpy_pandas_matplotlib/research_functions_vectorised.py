import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)

    return np.array_equal(unique_x, unique_y) and np.array_equal(counts_x, counts_y)


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    prods = x[:-1] * x[1:]
    indices = (x[:-1] % 3 == 0) | (x[1:] % 3 == 0)
    filtered_prods = prods[indices]

    return np.max(filtered_prods) if filtered_prods.size > 0 else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.dot(image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    expanded_x = np.repeat(x[:, 0], x[:, 1])
    expanded_y = np.repeat(y[:, 0], y[:, 1])

    if expanded_x.shape[0] == expanded_y.shape[0]:
        return np.sum(expanded_x * expanded_y)
    else:
        return -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    norm_x = np.linalg.norm(X, axis=1)
    norm_y = np.linalg.norm(Y, axis=1)

    norm_x[norm_x == 0] = 1
    norm_y[norm_y == 0] = 1

    distance_matrix = np.dot(X, Y.T) / np.dot(norm_x[:, np.newaxis], norm_y[np.newaxis, :])
    distance_matrix[np.all(distance_matrix == 0, axis=1)] = 1

    return distance_matrix