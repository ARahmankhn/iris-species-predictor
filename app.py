import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 5px solid #1e88e5;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and return the Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: iris.target_names[0], 
                                          1: iris.target_names[1], 
                                          2: iris.target_names[2]})
    return df, iris

def load_or_create_model():
    """Load existing model or create a new one"""
    try:
        with open('deployment.pkl', 'rb') as file:
            model_data = pickle.load(file)
        return model_data['model'], model_data['feature_names'], model_data['target_names']
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        st.warning(f"Could not load existing model: {e}")
        st.info("Creating a new model...")
        return create_new_model()

def create_new_model():
    """Create and train a new model"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save the model
    model_data = {
        'model': model,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }
    
    try:
        with open('deployment.pkl', 'wb') as file:
            pickle.dump(model_data, file)
        st.success("New model created and saved successfully!")
    except Exception as e:
        st.warning(f"Could not save model: {e}")
    
    return model, iris.feature_names, iris.target_names

def make_prediction(model, features):
    """Make prediction with confidence scores"""
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = np.max(probabilities) * 100
    
    return prediction, confidence, probabilities

def create_feature_importance_chart(model, feature_names):
    """Create feature importance visualization"""
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_df, 
        x='importance', 
        y='feature',
        orientation='h',
        title='Feature Importance in Decision Tree',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_species_distribution_chart(df):
    """Create species distribution pie chart"""
    species_counts = df['species_name'].value_counts()
    
    fig = px.pie(
        values=species_counts.values,
        names=species_counts.index,
        title='Iris Species Distribution in Dataset',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_scatter_matrix(df):
    """Create scatter plot matrix"""
    fig = px.scatter_matrix(
        df,
        dimensions=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        color='species_name',
        title='Iris Features Scatter Matrix',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(height=600)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Iris Species Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Predict the species of Iris flowers using machine learning")
    
    # Load data and model
    df, iris_data = load_data()
    model, feature_names, target_names = load_or_create_model()
    
    # Sidebar for input
    st.sidebar.header("🔧 Input Features")
    st.sidebar.markdown("Adjust the sliders to input flower measurements:")
    
    # Input sliders
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)", 
        min_value=float(df['sepal length (cm)'].min()), 
        max_value=float(df['sepal length (cm)'].max()), 
        value=5.0,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)", 
        min_value=float(df['sepal width (cm)'].min()), 
        max_value=float(df['sepal width (cm)'].max()), 
        value=3.0,
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.sidebar.slider(
        "Petal Length (cm)", 
        min_value=float(df['petal length (cm)'].min()), 
        max_value=float(df['petal length (cm)'].max()), 
        value=4.0,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.sidebar.slider(
        "Petal Width (cm)", 
        min_value=float(df['petal width (cm)'].min()), 
        max_value=float(df['petal width (cm)'].max()), 
        value=1.3,
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    # Create feature array
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Make prediction
    prediction, confidence, probabilities = make_prediction(model, features)
    predicted_species = target_names[prediction]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction result
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-result">🌺 {predicted_species.title()}</div>
            <div class="confidence-score">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Input summary
        st.markdown('<div class="sub-header">📊 Input Summary</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features,
            'Unit': ['cm'] * 4
        })
        st.dataframe(input_df, use_container_width=True)
    
    with col2:
        # Probability distribution
        st.markdown('<div class="sub-header">🎯 Prediction Probabilities</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            'Species': target_names,
            'Probability': probabilities * 100
        }).sort_values('Probability', ascending=False)
        
        fig_prob = px.bar(
            prob_df,
            x='Probability',
            y='Species',
            orientation='h',
            color='Probability',
            color_continuous_scale='RdYlBu_r',
            text='Probability'
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig_prob.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Probability (%)",
            yaxis_title="Species"
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Additional sections
    st.markdown("---")
    
    # Tabs for additional information
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Feature Importance", "🔍 Data Exploration", "📋 Dataset Info", "ℹ️ About"])
    
    with tab1:
        st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
        fig_importance = create_feature_importance_chart(model, feature_names)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Understanding Feature Importance:</strong><br>
        This chart shows which features are most important for the decision tree model's predictions. 
        Higher values indicate features that contribute more to distinguishing between iris species.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Dataset Exploration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Species distribution
            fig_dist = create_species_distribution_chart(df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Statistics
            st.markdown("**Dataset Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Scatter matrix
        st.markdown("**Feature Relationships:**")
        fig_scatter = create_scatter_matrix(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Dataset Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview:**")
            st.write(f"• **Total Samples:** {len(df)}")
            st.write(f"• **Features:** {len(feature_names)}")
            st.write(f"• **Classes:** {len(target_names)}")
            st.write(f"• **Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.markdown("**Species Count:**")
            for species in target_names:
                count = len(df[df['species_name'] == species])
                st.write(f"• **{species.title()}:** {count} samples")
        
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">About This Application</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>🌸 Iris Species Classifier</strong><br><br>
        
        This application uses a <strong>Decision Tree Classifier</strong> to predict the species of Iris flowers 
        based on four morphological features:
        
        <ul>
        <li><strong>Sepal Length:</strong> Length of the sepal in centimeters</li>
        <li><strong>Sepal Width:</strong> Width of the sepal in centimeters</li>
        <li><strong>Petal Length:</strong> Length of the petal in centimeters</li>
        <li><strong>Petal Width:</strong> Width of the petal in centimeters</li>
        </ul>
        
        <strong>The three Iris species are:</strong>
        <ul>
        <li><strong>Setosa:</strong> Generally has smaller petals and larger sepals</li>
        <li><strong>Versicolor:</strong> Has intermediate measurements</li>
        <li><strong>Virginica:</strong> Usually has the largest petals</li>
        </ul>
        
        <strong>Model Performance:</strong><br>
        The Decision Tree model achieves high accuracy on the Iris dataset, making it an excellent 
        example for demonstrating machine learning classification.
        
        <br><br>
        <strong>How to Use:</strong><br>
        1. Adjust the sliders in the sidebar to input flower measurements<br>
        2. The prediction and confidence will update automatically<br>
        3. Explore the additional tabs for more insights about the data and model
        </div>
        """, unsafe_allow_html=True)
        
        # Model information
        st.markdown("**Model Information:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• **Algorithm:** Decision Tree Classifier")
            st.write(f"• **Max Depth:** {model.max_depth}")
            st.write(f"• **Criterion:** {model.criterion}")
        
        with col2:
            st.write(f"• **Random State:** {model.random_state}")
            st.write(f"• **Features:** {model.n_features_in_}")
            st.write(f"• **Classes:** {model.n_classes_}")

if __name__ == "__main__":
    main()
