import numpy as np
import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        '''
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        '''
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.mapping = None

    def fit(self, data, labels):
        '''
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        '''
        cluster_labels = self.kmeans.fit_predict(data)
        self.mapping, _ = self._best_fit_classification(cluster_labels, labels)

        return self

    def predict(self, data):
        '''
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        '''
        cluster_preds = self.kmeans.predict(data)

        return np.array([self.mapping[cluster] for cluster in cluster_preds])

    def _best_fit_classification(self, cluster_labels, true_labels):
        '''
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        '''
        unique_clusters = np.arange(self.n_clusters)
        unique_classes = np.unique(true_labels[true_labels != -1])

        mapping = np.zeros(self.n_clusters, dtype=int)

        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            cluster_labels_in = true_labels[mask]

            labeled_mask = cluster_labels_in != -1
            labeled_classes = cluster_labels_in[labeled_mask]

            if len(labeled_classes) > 0:
                counts = np.bincount(labeled_classes)
                mapping[cluster] = np.argmax(counts)
            else:
                counts = np.bincount(true_labels[true_labels != -1])
                mapping[cluster] = np.argmax(counts)

        predicted_labels = np.array([mapping[cluster] for cluster in cluster_labels])

        return mapping, predicted_labels
