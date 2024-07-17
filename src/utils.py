import numpy as np  
from sklearn.model_selection import train_test_split


def loadong_data(data_PATH):
    data = np.loadtxt(data_PATH, delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(
        data[:, :-1], data[:, -1], test_size=0.2, random_state=26
    )
    return X_train, X_test, y_train, y_test

def standardization(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
