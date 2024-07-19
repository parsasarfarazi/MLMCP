from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
    
    lrm = LogisticRegression(solver='liblinear', random_state=0)
    startTrainTime = time.time()
    lrm.fit(X_train, y_train)
    training_time = time.time()-startTrainTime
    y_pred = lrm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, training_time

def Decision_tree(X_train, X_test, y_train, y_test):
    """
    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        
    Returns:
        Tuple: (y_pred, accuracy, training_time)
    """
    
    clf = DecisionTreeClassifier(criterion= 'entropy', splitter='best', max_depth=15, min_samples_split=2,
                             min_samples_leaf=1 , )
    startTrainTime=time.time()
    clf.fit(X_train, y_train)
    training_time = time.time()-startTrainTime
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, training_time

def Random_forest(X_train, X_test, y_train, y_test):
    """
    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        
    Returns:
        Tuple: (y_pred, accuracy, training_time)
    """
    
    clrf = RandomForestClassifier(n_estimators=10, criterion= 'entropy', max_depth=20, min_samples_split=2,
                             min_samples_leaf=1 , )
    startTrainTime=time.time()
    clrf.fit(X_train, y_train)
    training_time = time.time()-startTrainTime
    y_pred = clrf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, training_time

def Neural_network(X_train, X_test, y_train, y_test):
    """
    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        
    Returns:
        Tuple: (y_pred, accuracy, training_time)
    """
    
    NN = MLPClassifier(hidden_layer_sizes=(50,25),  
                      activation='relu',          
                      solver='adam',              
                      max_iter=500,              
                      random_state=41) 
    startTrainTime=time.time()
    NN.fit(X_train, y_train)
    training_time = time.time()-startTrainTime
    y_pred = NN.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy, training_time
