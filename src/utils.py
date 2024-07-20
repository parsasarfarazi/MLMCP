import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

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

# In Progress
# def filter_data(X_train, X_test):
#     """_summary_

#     Args:
#         model (_type_): _description_
#         X_train (_type_): _description_
#         X_test (_type_): _description_
#     Return:
#         tuple(filtered_X_train, filtered_X_test)
#     """
    
#     weighs_num = model.coef_.shape[1]
#     weighs = model.coef_
#     indexes = []
#     for i in range(weighs_num):
#         if abs(weighs[0][i])<0.1:
#             indexes.append(i)
#     filtered_X_train = np.delete(X_train, indexes, axis=1)
#     filtered_X_test = np.delete(X_test, indexes, axis=1)
#     return filtered_X_train, filtered_X_test

def perform_reduction(reduction_method, X_test):
    if reduction_method == "PCA":
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_test)
    elif reduction_method == "TSNE":
        tsne = TSNE(n_components=2, random_state=42)
        X_reduced = tsne.fit_transform(X_test)
    return X_reduced

def show_plot(X_reduced, y_pred):
    """

    Args:
        X_reduced (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        Figure: _description_
    """
    plt.figure(figsize=(12, 8)) 
    plt.scatter(X_reduced[y_pred == True, 0], X_reduced[y_pred == True, 1], c='green', label='Spam', s=10)  # Green for True predictions
    plt.scatter(X_reduced[y_pred == False, 0], X_reduced[y_pred == False, 1], c='red', label='Not spam', s=10)  # Red for False predictions

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Model Predictions (Reduced Dimensions)")
    plt.legend()  
    
    return plt