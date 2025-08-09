import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set the page layout and title
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

# Add a header with some padding
st.title("🌸 **Iris Flower Species Prediction** 🌸")
st.markdown("""
    <h3 style='text-align: center; color: #2E3B55;'>Enter the flower measurements below and press "Predict" to see the predicted species</h3>
""", unsafe_allow_html=True)

# Load the model and feature names
@st.cache_resource
def load_model():
    """
    Loads the trained machine learning model from a pickle file.
    Uses st.cache_resource to avoid reloading the model on every rerun.
    Includes a try/except block for graceful failure if the file is not found.
    """
    try:
        model_bundle = joblib.load("deployment.pkl")
        return model_bundle
    except FileNotFoundError:
        st.error("Model file 'deployment.pkl' not found. Please ensure it's in the same directory.")
        return None

# Load the model and check if it was successful
model_bundle = load_model()
if model_bundle:
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    target_names = model_bundle["target_names"]

    # Add instructions for the user
    st.markdown("""
        <p style='text-align: center; color: #4F7D89;'>This app predicts the species of an Iris flower based on its <b>petal length</b>, <b>petal width</b>, <b>sepal length</b>, and <b>sepal width</b>. Please enter values for all features.</p>
    """, unsafe_allow_html=True)

    # Layout: Use two columns for better organization
    col1, col2 = st.columns(2)

    # Input fields with placeholders and tooltips for better user experience
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Length of the sepal in centimeters")
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, help="Length of the petal in centimeters")

    with col2:
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, help="Width of the sepal in centimeters")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Width of the petal in centimeters")

    # Add a separator for better organization
    st.markdown("---")

    # Loading spinner when prediction is happening
    with st.spinner("Making the prediction..."):
        if st.button("🚀 Predict"):
            # Prepare the input for the model
            input_data = np.array([[sepal_length, petal_length, sepal_width, petal_width]])

            # Prediction
            prediction = model.predict(input_data)[0]
            predicted_species = target_names[int(prediction)]
            
            # Displaying prediction with a nice heading and color
            st.markdown(f"<h2 style='text-align: center; color: #28a745;'>**Predicted Species: {predicted_species}**</h2>", unsafe_allow_html=True)

            # Display probabilities if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                
                # Ensure target_names is aligned with proba length
                if len(proba) == len(target_names):
                    df = pd.DataFrame({"Species": target_names, "Probability": proba})
                    st.subheader("Prediction Probabilities")
                    st.bar_chart(df.set_index("Species"))
                else:
                    st.error("Error: Mismatch between number of classes and prediction probabilities.")
            
            st.success("Prediction Complete!")

    # Display footer or credits if required
    st.markdown("""
        <footer style='text-align: center; color: gray; font-size: 12px;'>
            Iris Species Prediction App created by <b>Your Name</b>. Powered by Streamlit.
        </footer>
    """, unsafe_allow_html=True)
