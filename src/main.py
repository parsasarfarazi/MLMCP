import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from utils import loadong_data, standardization
from models import Logistic_regression
import streamlit as st

# Streamlit App
st.set_page_config(layout="wide")

# Navigation Bar
page = st.sidebar.radio("Select a Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"])

X_train, X_test, y_train, y_test = loadong_data("spambase.data")
X_train, X_test = standardization(X_train, X_test)

# Page Content
if page == "Logistic Regression":
    st.title("Logistic Regression")
    st.write("This page demonstrates Logistic Regression.")
    
    # Buttons
    run_original = st.button("Run on Original Data")
    run_filtered = st.button("Run on Filtered Data")

    if run_original:
        y_pred, accuracy,training_time = Logistic_regression(X_train, X_test, y_train, y_test )
        st.write(f"Accuracy: {accuracy:.2f}")
        
        # Dimensionality Reduction (PCA or TSNE)
        # Choose a dimensionality reduction method (PCA or TSNE)
        reduction_method = st.selectbox("Dimensionality Reduction Method", ["PCA", "TSNE"])

        if reduction_method == "PCA":
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_test)
        elif reduction_method == "TSNE":
            tsne = TSNE(n_components=2, random_state=42)
            X_reduced = tsne.fit_transform(X_test)


        plt.scatter(X_reduced[y_pred == True, 0], X_reduced[y_pred == True, 1], c='green', label='Spam', s=10)  # Green for True predictions
        plt.scatter(X_reduced[y_pred == False, 0], X_reduced[y_pred == False, 1], c='red', label='Not spam', s=10)  # Red for False predictions

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Model Predictions (Reduced Dimensions)")
        plt.legend()  # Show the legend
        st.pyplot(plt)
        
        
    if run_filtered:
        # Implement filtering logic and pass filtered data to logistic_regression()
        # ...
        st.write("Filtered data results will be displayed here.")

elif page == "Decision Tree":
    st.title("Decision Tree")
    st.write("This page demonstrates Decision Tree.")
    
    # Buttons
    run_original = st.button("Run on Original Data")
    run_filtered = st.button("Run on Filtered Data")

    if run_original:
        model, accuracy, X_test, y_test, y_pred = decision_tree(data)
        st.write(f"Accuracy: {accuracy:.2f}")

        # Visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test["feature1"], X_test["feature2"], c=y_pred)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Tree Predictions")
        st.pyplot(plt)

    if run_filtered:
        # Implement filtering logic and pass filtered data to decision_tree()
        # ...
        st.write("Filtered data results will be displayed here.")

# elif page == "Random Fore



