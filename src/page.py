import streamlit as st
from utils import show_plot, perform_reduction
from models import *

def show_page(model_type, X_train, X_test, y_train, y_test ):
    """

    Args:
        model_type (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    if  model_type=="Logistic Regression":
        model = Logistic_regression
    elif model_type == "Decision Tree":
        model = Decision_tree
    elif  model_type == "Random Forest":
        model = Random_forest
    elif model_type == "Neural Network":
        model = Neural_network
    
    st.title(model_type)
    st.write(f"This page demonstrates {model_type}.")
    
    run_original = st.button("Run on Original Data")
    run_filtered = st.button("Run on Filtered Data")

    if run_original:
        
        y_pred, accuracy, training_time = model(X_train, X_test, y_train, y_test )
        st.write(f"**Accuracy:** {accuracy*100:.0f}%") 
        st.write(f"**Training Time**: {training_time:.4f} seconds")
        X_reduced = perform_reduction("TSNE", X_test)
        fig = show_plot(X_reduced,y_pred)
        st.pyplot(fig,use_container_width=True)
        
        
    if run_filtered:

        st.write("Filtered data results will be displayed here.")
        st.write("**In Progress...**")