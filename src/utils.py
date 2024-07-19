import numpy as np  
from sklearn.model_selection import train_test_split


def loadong_data(data_PATH):
    """

    Args:
        data_PATH (_type_): _description_

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    data = np.loadtxt(data_PATH, delimiter=',')
    X_train, X_test, y_train, y_test = train_test_split(
        data[:, :-1], data[:, -1], test_size=0.2, random_state=26
    )
    return X_train, X_test, y_train, y_test

def standardization(X_train, X_test):
    """

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_

    Returns:
        _type_: X_train, X_test
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test

def filter_data(model, X_train, X_test):
    """_summary_

    Args:
        model (_type_): _description_
        X_train (_type_): _description_
        Xtest (_type_): _description_
    Return:

    """
    weighs_num = model.coef_.shape[1]
    weighs = model.coef_
    indexes = []
    for i in range(weighs_num):
        if abs(weighs[0][i])<0.1:
            indexes.append(i)
    filtered_X_train = np.delete(X_train, indexes, axis=1)
    filtered_X_test = np.delete(X_test, indexes, axis=1)
    return filtered_X_train, filtered_X_test