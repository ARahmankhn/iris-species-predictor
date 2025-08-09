import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set the page layout and title
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

# Add a header with some padding
st.title("🌸 **Iris Flower Species Prediction** 🌸")
…            
            st.success("Prediction Complete!")

    # Display footer or credits if required
    st.markdown("""
        <footer style='text-align: center; color: gray; font-size: 12px;'>
            Iris Species Prediction App created by <b>Your Name</b>. Powered by Streamlit.
        </footer>
    """, unsafe_allow_html=True)
