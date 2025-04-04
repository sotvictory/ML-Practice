import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
from numpy import ndarray

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    categorical_columns = ["genres", "directors", "filming_locations"]

    for category in categorical_columns:
        df_train[category] = df_train[category].apply(
            lambda x: "unknown" if x == "unknown" else ",".join(x)
        )
        df_test[category] = df_test[category].apply(
            lambda x: "unknown" if x == "unknown" else ",".join(x)
        )

    y_train = df_train["awards"]
    X_train = df_train.drop(["awards", "keywords"], axis=1)
    X_test = df_test.drop(["keywords"], axis=1)

    vectorizer = CountVectorizer()

    for column in categorical_columns:
        train_vec = vectorizer.fit_transform(X_train[column]).toarray()
        test_vec = vectorizer.transform(X_test[column]).toarray()

        cols = [elem + '_' + column for elem in vectorizer.get_feature_names_out()]
        X_train = pd.concat([X_train, pd.DataFrame(train_vec, columns=cols)], axis=1)
        X_test = pd.concat([X_test, pd.DataFrame(test_vec, columns=cols)], axis=1)
        X_train = X_train.drop([column], axis=1)
        X_test = X_test.drop([column], axis=1)

    genders = ["actor_0_gender", "actor_1_gender", "actor_2_gender"]

    X_train[genders] = X_train[genders].astype("category")
    X_test[genders] = X_test[genders].astype("category")

    model = CatBoostRegressor(
        n_estimators=1679,
        max_depth=5,
        learning_rate=0.027189176153113746,
        verbose=0,
        train_dir='/tmp/catboost_info'
    )

    model.fit(X_train, y_train, cat_features=genders)

    return model.predict(X_test.to_numpy())
