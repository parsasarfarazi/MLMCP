from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time


def Logistic_regression(X_train, X_test, y_train, y_test):
    """
    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        
    Returns:
        Tuple: (y_pred, accuracy, training_time)
    """
    
    startTrainTime = time.time()
    lrm = LogisticRegression(solver='liblinear', random_state=0)
    lrm.fit(X_train, y_train)
    training_time = time.time()-startTrainTime
    y_pred = lrm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, training_time