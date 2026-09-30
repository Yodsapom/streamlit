import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for Teal/Cyan Theme
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #EBF5F6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #D3E8EA;
    }
    
    /* Custom Headers & Icons */
    h1 {
        color: #0E5A60;
    }
    h2, h3 {
        color: #12636B;
    }
    
    /* Primary Accent Buttons */
    .stButton>button {
        background-color: #12636B;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0E5A60;
        color: white;
    }
    
    /* Result Card Styling */
    .result-card {
        background: linear-gradient(135deg, #0D5C75 0%, #197B82 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 18px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    .result-value {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-confidence {
        font-size: 14px;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

# Load Dataset & Train Model
@st.cache_data
def load_data_and_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Calculate feature averages
    avg_df = X.mean().reset_index()
    avg_df.columns = ['Feature', 'Dataset Average']
    
    return iris, model, avg_df

iris, model, avg_df = load_data_and_model()

# Header Section
st.markdown("<h1 style='display: flex; align-items: center; gap: 10px;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.write("Predict the species of Iris flowers using Machine Learning")

st.markdown("---")

# Sidebar - Input Features
st.sidebar.markdown("### 📊 Input Features")
st.sidebar.caption("Adjust the sliders to input flower measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

predict_btn = st.sidebar.button("Predict Species")

# Layout: Split into 2 Columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📈 Input Visualization")
    st.caption("Your Input vs Dataset Average")
    
    user_inputs = [sepal_length, sepal_width, petal_length, petal_width]
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    dataset_averages = avg_df['Dataset Average'].values
    
    # Bar Chart with Teal Theme
    fig = go.Figure(data=[
        go.Bar(name='Your Input', x=features, y=user_inputs, marker_color='#2A9D8F', text=user_inputs, textposition='auto'),
        go.Bar(name='Dataset Average', x=features, y=dataset_averages, marker_color='#264653', text=np.round(dataset_averages, 2), textposition='auto')
    ])
    
    fig.update_layout(
        barmode='group',
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Prediction Result")
    
    # Model Prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction_idx = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    predicted_species = iris.target_names[prediction_idx].capitalize()
    confidence = probabilities[prediction_idx] * 100
    
    # Prediction Card UI
    st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Predicted Species</div>
            <div class="result-value">{predicted_species}</div>
            <div class="result-confidence">Confidence: {confidence:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Probability Distribution")
    
    # Probability Bar Chart
    species_names = [name.capitalize() for name in iris.target_names]
    bar_colors = ['#006666' if i == prediction_idx else '#80B3B3' for i in range(len(species_names))]
    
    fig_prob = go.Figure(data=[
        go.Bar(
            x=species_names,
            y=probabilities * 100,
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig_prob.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis=dict(title='Probability (%)', range=[0, 100]),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_prob, use_container_width=True)