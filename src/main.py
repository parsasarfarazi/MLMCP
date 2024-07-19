import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from utils import *
from page import show_page
import streamlit as st


st.set_page_config(layout="wide")

page = st.sidebar.radio("Select a Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"])

X_train, X_test, y_train, y_test = loadong_data("spambase.data")
X_train, X_test = standardization(X_train, X_test)

if page == "Logistic Regression":
    show_page("Logistic Regression",X_train, X_test, y_train, y_test)

elif page == "Decision Tree":
    show_page("Decision Tree",X_train, X_test, y_train, y_test)

elif page == "Random Forest":
    show_page("Random Forest",X_train, X_test, y_train, y_test)

elif page == "Neural Network":
    show_page("Neural Network",X_train, X_test, y_train, y_test)


