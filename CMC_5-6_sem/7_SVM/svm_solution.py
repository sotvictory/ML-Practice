import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    model = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced')
    model.fit(train_features_scaled, train_target)

    predictions = model.predict(test_features_scaled)

    return predictions
