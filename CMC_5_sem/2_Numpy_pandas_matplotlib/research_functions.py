from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x) == Counter(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_prod = -1
    
    for i in range(len(x) - 1):
        prod = x[i] * x[i + 1]
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            max_prod = max(max_prod, prod)
    
    return max_prod


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]

    for i in range(len(image)):
        for j in range(len(image[0])):
            res[i][j] = sum(image[i][j][k] * weights[k] for k in range(len(weights)))

    return res

def decode_rle(rle: List[List[int]]) -> List[int]:
    """
    Декодировать вектор, заданный в формате RLE.
    """
    decoded = []
    for value, count in rle:
        decoded.extend([value] * count)

    return decoded

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    decoded_x = decode_rle(x)
    decoded_y = decode_rle(y)
    if len(decoded_x) != len(decoded_y):
        return -1
    scalar_prod = sum(a * b for a, b in zip(decoded_x, decoded_y))
    
    return scalar_prod


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    distance_matrix = [[0] * len(Y) for _ in range(len(X))]  
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            dot_prod = sum(a * b for a, b in zip(x, y))
            
            norm_x = sum(a ** 2 for a in x) ** 0.5
            norm_y = sum(b ** 2 for b in y) ** 0.5
            
            if norm_x == 0 or norm_y == 0:
                distance_matrix[i][j] = 1.0
            else:
                distance_matrix[i][j] = (dot_prod / (norm_x * norm_y))
                
    return distance_matrix