import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
import sys


def Kmeans_Ridge_KNN(X, y, train_index, test_index, p1, p2, alpha=0.01, n_clusters=2):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    n_clusters = n_clusters

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, tol=1e-4)
    kmeans.fit(X_train)

    models = []
    y_train_pred = np.array([])
    y_train_true = np.array([])
    labels = kmeans.labels_
    idtrain = np.array([])
    for c in range(n_clusters):
        r = Ridge(alpha=alpha, normalize=False, solver="lsqr")
        r.fit(X_train[labels == c], y_train[labels == c])
        result = r.predict(X_train[labels == c])
        y_train_pred = np.append(y_train_pred, r.predict(X_train[labels == c]))
        y_train_true = np.append(y_train_true, y_train[labels == c])
        idtrain = np.append(idtrain, train_index[labels == c])
        models.append(r)

    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    knn.fit(X_train, kmeans.labels_)

    y_test_pred = np.array([])
    y_test_true = np.array([])
    labels = knn.predict(X_test)
    idtest = np.array([])
    for c in range(n_clusters):
        y_test_pred = np.append(y_test_pred, models[c].predict(X_test[labels == c]))
        y_test_true = np.append(y_test_true, y_test[labels == c])
        idtest = np.append(idtest, test_index[labels == c])

    for i in range(len(idtrain)):
        y_train_pred[i] *= (
            1.00
            + p1 * df.loc[idtrain[i]]["CHAS"] / 100
            + p2 * df.loc[idtrain[i]]["RAD"] / 100
        )
        y_train_true[i] *= (
            1.00
            + p1 * df.loc[idtrain[i]]["CHAS"] / 100
            + p2 * df.loc[idtrain[i]]["RAD"] / 100
        )

    for i in range(len(idtest)):
        y_test_pred[i] *= (
            1.00
            + p1 * df.loc[idtest[i]]["CHAS"] / 100
            + p2 * df.loc[idtest[i]]["RAD"] / 100
        )
        y_test_true[i] *= (
            1.00
            + p1 * df.loc[idtest[i]]["CHAS"] / 100
            + p2 * df.loc[idtest[i]]["RAD"] / 100
        )

    r2_on_train = r2_score(y_train_pred, y_train_true)
    r2_on_test = r2_score(y_test_pred, y_test_true)
    return r2_on_train, r2_on_test


def kfold(p1, p2):
    scaler = MinMaxScaler()
    X = df.drop(columns=["MEDV", "CHAS", "RAD"])
    X["ONES"] = [1] * X.shape[0]
    print(X.columns)
    cols = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    y = df["MEDV"] / (1.00 + df["CHAS"] * p1 / 100 + df["RAD"] * p2 / 100)
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    r2_train = []
    r2_test = []
    for train_index, test_index in kfold.split(X):
        r2_on_train, r2_on_test = Kmeans_Ridge_KNN(
            X, y, train_index, test_index, p1, p2
        )
        r2_train.append(r2_on_train)
        r2_test.append(r2_on_test)
    print(np.mean(r2_train), np.mean(r2_test))
    return np.mean(r2_train), np.mean(r2_test)


if __name__ == "__main__":
    df = pd.read_csv("processed_original.csv")
    print(df.head(10))
    best_p1 = best_p2 = best_r2_train = best_r2_test = 0
    prev_r2_train = prev_r2_test = 0
    for i in range(0, 5):
        for j in range(0, 5):
            p1 = i * 1.0
            p2 = j * 1.0
            print("Processing hypothesis with p1 =", p1, "p2 = ", p2)
            r2_train, r2_test = kfold(p1, p2)
            prev_r2_train, prev_r2_test = r2_train, r2_test
            if best_r2_test < r2_test:
                best_p1 = p1
                best_p2 = p2
                best_r2_train = r2_train
                best_r2_test = r2_test
    print(
        "Optimize p for hypothesis: p1 =",
        best_p1,
        "p2 =",
        best_p2,
        "r2_train:",
        best_r2_train,
        "r2_test:",
        best_r2_test,
    )
