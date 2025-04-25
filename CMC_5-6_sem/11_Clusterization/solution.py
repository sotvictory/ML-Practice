import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """
    labels = np.array(labels)
    n = len(labels)

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)

    # 1 кластер => 0
    if n_clusters == 1:
        return 0.0

    # D[i, j] — расстояние между объектом i и объектом j
    D = pairwise_distances(x)

    # mask[i, k] — true, если объект i принадлежит кластеру k
    mask = np.zeros((n, n_clusters), dtype=bool)
    mask[np.arange(n), inverse] = True

    # количество объектов в каждом кластере
    cluster_sizes = mask.sum(axis=0)

    # s[i] — среднее расстояние объекта i до других объектов своего кластера
    sum_intra = (D * mask.T[inverse]).sum(axis=1)
    sizes = cluster_sizes[inverse]
    s = np.zeros(n)
    mask_single = sizes == 1
    mask_multi = ~mask_single
    s[mask_multi] = sum_intra[mask_multi] / (sizes[mask_multi] - 1)
    s[mask_single] = 0.0

    # d[i] — минимальное среднее расстояние объекта i до ближайшего другого кластера
    avg_dist_to_clusters = D @ mask / cluster_sizes
    avg_dist_to_clusters[np.arange(n), inverse] = np.inf
    d = np.min(avg_dist_to_clusters, axis=1)

    # sil[i] — силуэт для объекта i
    sil = np.zeros(n)
    mask_single_cluster = cluster_sizes[inverse] == 1
    sil[mask_single_cluster] = 0
    mask_zero = (s == 0) & (d == 0) & ~mask_single_cluster
    mask_nonzero = ~mask_zero & ~mask_single_cluster
    sil[mask_zero] = 0.0
    sil[mask_nonzero] = (d[mask_nonzero] - s[mask_nonzero]) / np.maximum(
        d[mask_nonzero], s[mask_nonzero]
    )

    return np.mean(sil)


def bcubed_score(true_labels, predicted_labels):
    """
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    n = len(true_labels)

    # C(i) == C(j), L(i) == L(j)
    # correctness_matrix[i, j] == True: объекты i и j корректно сгруппированы вместе
    correctness_matrix = np.equal.outer(
        predicted_labels, predicted_labels
    ) & np.equal.outer(true_labels, true_labels)

    # вычисление precision-bcubed для каждого объекта
    cluster_sizes = np.sum(np.equal.outer(predicted_labels, predicted_labels), axis=1)
    precision_bcubed_per_object = np.sum(correctness_matrix, axis=1) / cluster_sizes

    # вычисление recall-bcubed для каждого объекта
    gold_sizes = np.sum(np.equal.outer(true_labels, true_labels), axis=1)
    recall_bcubed_per_object = np.sum(correctness_matrix, axis=0) / gold_sizes

    # их усреднение
    precision_bcubed = np.mean(precision_bcubed_per_object)
    recall_bcubed = np.mean(recall_bcubed_per_object)

    if precision_bcubed + recall_bcubed == 0:
        return 0.0

    # F_1-bcubed
    return 2 * (precision_bcubed * recall_bcubed) / (precision_bcubed + recall_bcubed)
