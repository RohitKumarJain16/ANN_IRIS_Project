import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Set matplotlib and seaborn style
plt.style.use('default')
sns.set_palette("husl")

st.set_page_config(
    page_title="Iris ANN Classifier",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #2E86AB !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.8rem;
        color: #F18F01 !important;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card h2 {
        color: white !important;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .metric-card h3 {
        color: #E8F4FD !important;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 500;
    }
    
    .metric-card p {
        color: #E8F4FD !important;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .prediction-result h1 {
        color: white !important;
        font-size: 3rem;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-result h2, .prediction-result h3 {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #2c3e50 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-box h3 {
        color: #d35400 !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .info-box p, .info-box li {
        color: #2c3e50 !important;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .info-box strong {
        color: #c0392b !important;
    }
    
    /* Warning box styling */
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #721c24 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box h3 {
        color: #a94442 !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .warning-box p {
        color: #721c24 !important;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Success box for additional styling */
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #155724 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Slider improvements */
    .stSlider > div > div > div {
        color: #2E86AB !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the iris dataset"""
    try:
        df = pd.read_csv('iris.csv')
        return df
    except FileNotFoundError:
        st.error("❌ iris.csv file not found. Please ensure the file is in the same directory as app.py")
        return None

def create_and_train_model(X_train_scaled, y_train_cat):
    """Create and train a new model"""
    model = Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with st.spinner('🔄 Training the model... This may take a few moments.'):
        progress_bar = st.progress(0)
        history = model.fit(
            X_train_scaled, y_train_cat, 
            epochs=100, 
            batch_size=8, 
            validation_split=0.2, 
            verbose=0
        )
        progress_bar.progress(100)
    
    # Save the model
    model.save('iris_ann_model.h5')
    st.success('✅ Model trained and saved successfully!')
    
    return model, history

@st.cache_resource
def load_or_create_model(X_train_scaled, y_train_cat):
    """Load existing model or create new one if not found"""
    if os.path.exists('iris_ann_model.h5'):
        try:
            model = load_model('iris_ann_model.h5')
            st.success("✅ Existing model loaded successfully!")
            return model, None
        except Exception as e:
            st.warning(f"⚠️ Error loading existing model: {str(e)}")
            st.info("🔄 Creating new model...")
            return create_and_train_model(X_train_scaled, y_train_cat)
    else:
        st.warning("⚠️ Model file not found. Creating and training new model...")
        return create_and_train_model(X_train_scaled, y_train_cat)

@st.cache_data
def prepare_data(df):
    """Prepare data for training and prediction"""
    # Separate features and target
    X = df.drop(columns=['variety'])
    y = df['variety']
    
    # Label encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encoding for training
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat, scaler, encoder

def create_matplotlib_plots(df):
    """Create matplotlib/seaborn plots"""
    # Set up the matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis with Matplotlib & Seaborn', fontsize=16, fontweight='bold')
    
    # Plot 1: Seaborn pairplot
    features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    
    # Plot 2: Box plots
    df_melted = df.melt(id_vars=['variety'], value_vars=features)
    sns.boxplot(data=df_melted, x='variable', y='value', hue='variety', ax=axes[0, 0])
    axes[0, 0].set_title('Feature Distributions by Species')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 3: Correlation heatmap
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Feature Correlation Heatmap')
    
    # Plot 4: Violin plots
    sns.violinplot(data=df_melted, x='variable', y='value', hue='variety', ax=axes[1, 0])
    axes[1, 0].set_title('Feature Density Distributions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Scatter plot matrix (just one pair)
    sns.scatterplot(data=df, x='sepal.length', y='sepal.width', hue='variety', 
                   s=100, alpha=0.7, ax=axes[1, 1])
    axes[1, 1].set_title('Sepal Length vs Sepal Width')
    
    plt.tight_layout()
    return fig

def create_seaborn_pairplot(df):
    """Create seaborn pair plot"""
    # Create pairplot
    pair_fig = plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df, hue='variety', diag_kind='hist', 
                           plot_kws={'alpha': 0.7, 's': 80},
                           diag_kws={'alpha': 0.7})
    pairplot.fig.suptitle('Iris Dataset Pairplot with Seaborn', y=1.02, fontsize=16)
    return pairplot.fig

def create_training_plots(history):
    """Create training history plots using matplotlib"""
    if history is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#2E86AB', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#F18F01', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss', color='#C73E1D', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#A8A8A8', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pairplot(df):
    """Create an interactive pairplot using plotly"""
    features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[f"{features[i]} vs {features[j]}" for i in range(4) for j in range(4)]
    )
    
    # Updated color scheme for better visibility
    colors = {'Setosa': '#2E86AB', 'Versicolor': '#F18F01', 'Virginica': '#C73E1D'}
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            for variety in df['variety'].unique():
                subset = df[df['variety'] == variety]
                fig.add_trace(
                    go.Scatter(
                        x=subset[feat2],
                        y=subset[feat1],
                        mode='markers',
                        name=variety,
                        marker=dict(color=colors[variety], size=8, opacity=0.7),
                        showlegend=(i == 0 and j == 0)
                    ),
                    row=i+1, col=j+1
                )
    
    fig.update_layout(
        height=800, 
        title="Iris Dataset Feature Relationships",
        title_font_size=20,
        title_font_color="#2E86AB"
    )
    return fig

def predict_species(model, scaler, encoder, sepal_length, sepal_width, petal_length, petal_width):
    """Make prediction for new input"""
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict(input_scaled, verbose=0)
    prediction_class = np.argmax(prediction_proba, axis=1)[0]
    
    # Convert back to species name
    species_name = encoder.inverse_transform([prediction_class])[0]
    confidence = prediction_proba[0][prediction_class] * 100
    
    return species_name, confidence, prediction_proba[0]

def main():
    # Header with improved styling
    st.markdown('<h1 class="main-header">🌺 Iris Species Classification with ANN</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📈 Data Analysis", "📊 Matplotlib & Seaborn", "🔮 Make Prediction", "📊 Model Performance", "ℹ️ About"]
    )
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, y_train_cat, y_test_cat, scaler, encoder = prepare_data(df)
    
    # Load or create model
    model, history = load_or_create_model(X_train_scaled, y_train_cat)
    
    if model is None:
        st.error("❌ Failed to load or create model. Please check your setup.")
        st.stop()
    
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
                    caption="Iris Flower", width=400)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This application demonstrates an Artificial Neural Network (ANN) for classifying Iris flowers into three species:</p>
        <ul>
        <li><strong>Setosa</strong> - Easily distinguishable species</li>
        <li><strong>Versicolor</strong> - Medium complexity classification</li>
        <li><strong>Virginica</strong> - Similar to Versicolor, requires careful analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model file exists
        if not os.path.exists('iris_ann_model.h5'):
            st.markdown("""
            <div class="warning-box">
            <h3>⚠️ Model Status</h3>
            <p>Model was created and trained during this session. The model file has been saved as 'iris_ann_model.h5' for future use.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset overview with improved styling
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <h2>150</h2>
            <p>samples</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>🔢 Features</h3>
            <h2>4</h2>
            <p>measurements</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3>🎯 Classes</h3>
            <h2>3</h2>
            <p>species</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h3>⚖️ Balance</h3>
            <h2>50</h2>
            <p>each class</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<h2 class="sub-header">📋 Dataset Quick View</h2>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Class distribution with updated colors
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.bar(
                df['variety'].value_counts().reset_index(),
                x='variety', y='count',
                title="Class Distribution",
                color='variety',
                color_discrete_map={'Setosa': '#2E86AB', 'Versicolor': '#F18F01', 'Virginica': '#C73E1D'}
            )
            fig_dist.update_layout(title_font_color="#2E86AB", title_font_size=16)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                df, names='variety',
                title="Species Distribution",
                color_discrete_map={'Setosa': '#2E86AB', 'Versicolor': '#F18F01', 'Virginica': '#C73E1D'}
            )
            fig_pie.update_layout(title_font_color="#2E86AB", title_font_size=16)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    elif page == "📈 Data Analysis":
        st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis (Plotly)</h2>', unsafe_allow_html=True)
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Feature relationships
        st.markdown("### 🔗 Interactive Feature Relationships")
        fig_pair = create_pairplot(df)
        st.plotly_chart(fig_pair, use_container_width=True)
        
        # Correlation heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌡️ Feature Correlation")
            numeric_df = df.select_dtypes(include=[np.number])
            fig_corr = px.imshow(
                numeric_df.corr(),
                text_auto=True,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            fig_corr.update_layout(title_font_color="#2E86AB", title_font_size=16)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### 📏 Feature Distributions")
            feature = st.selectbox("Select feature:", numeric_df.columns)
            fig_hist = px.histogram(
                df, x=feature, color='variety',
                title=f"{feature} Distribution by Species",
                marginal="box",
                color_discrete_map={'Setosa': '#2E86AB', 'Versicolor': '#F18F01', 'Virginica': '#C73E1D'}
            )
            fig_hist.update_layout(title_font_color="#2E86AB", title_font_size=16)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    elif page == "📊 Matplotlib & Seaborn":
        st.markdown('<h2 class="sub-header">📊 Data Analysis with Matplotlib & Seaborn</h2>', unsafe_allow_html=True)
        
        # Create matplotlib plots
        st.markdown("### 📊 Comprehensive Data Visualization")
        matplotlib_fig = create_matplotlib_plots(df)
        st.pyplot(matplotlib_fig)
        
        st.markdown("---")
        
        # Create seaborn pairplot
        st.markdown("### 🎨 Seaborn Pairplot")
        pairplot_fig = create_seaborn_pairplot(df)
        st.pyplot(pairplot_fig)
        
        st.markdown("---")
        
        # Additional matplotlib visualizations
        st.markdown("### 📈 Additional Statistical Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plots
            fig, ax = plt.subplots(figsize=(10, 6))
            for species in df['variety'].unique():
                subset = df[df['variety'] == species]
                ax.hist(subset['petal.length'], alpha=0.7, label=species, bins=15)
            ax.set_xlabel('Petal Length (cm)')
            ax.set_ylabel('Frequency')
            ax.set_title('Petal Length Distribution by Species')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Box plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='petal.width', by='variety', ax=ax)
            ax.set_title('Petal Width by Species')
            ax.set_xlabel('Species')
            ax.set_ylabel('Petal Width (cm)')
            plt.suptitle('')  # Remove the automatic title
            st.pyplot(fig)
    
    elif page == "🔮 Make Prediction":
        st.markdown('<h2 class="sub-header">🔮 Predict Iris Species</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Enter Flower Measurements")
            
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0, max_value=8.0, value=5.5, step=0.1,
                help="Length of the sepal in centimeters"
            )
            
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0, max_value=5.0, value=3.0, step=0.1,
                help="Width of the sepal in centimeters"
            )
            
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0, max_value=7.0, value=4.0, step=0.1,
                help="Length of the petal in centimeters"
            )
            
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1, max_value=3.0, value=1.5, step=0.1,
                help="Width of the petal in centimeters"
            )
            
            if st.button("🎯 Predict Species", type="primary"):
                species, confidence, proba = predict_species(
                    model, scaler, encoder, sepal_length, sepal_width, petal_length, petal_width
                )
                
                st.session_state.prediction_made = True
                st.session_state.species = species
                st.session_state.confidence = confidence
                st.session_state.proba = proba
        
        with col2:
            if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
                st.markdown(f"""
                <div class="prediction-result">
                <h2>🎉 Prediction Result</h2>
                <h1>{st.session_state.species}</h1>
                <h3>Confidence: {st.session_state.confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution with updated colors (Plotly)
                species_names = ['Setosa', 'Versicolor', 'Virginica']
                colors = ['#2E86AB', '#F18F01', '#C73E1D']
                fig_proba = go.Figure(data=[
                    go.Bar(
                        x=species_names,
                        y=st.session_state.proba * 100,
                        marker_color=colors
                    )
                ])
                fig_proba.update_layout(
                    title="Prediction Probabilities",
                    xaxis_title="Species",
                    yaxis_title="Probability (%)",
                    showlegend=False,
                    title_font_color="#2E86AB",
                    title_font_size=16
                )
                st.plotly_chart(fig_proba, use_container_width=True)
                
                # Alternative matplotlib visualization
                st.markdown("### 📊 Matplotlib Probability Chart")
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(species_names, st.session_state.proba * 100, color=colors, alpha=0.8)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Prediction Probabilities (Matplotlib)')
                ax.set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, prob in zip(bars, st.session_state.proba * 100):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            else:
                st.info("👆 Adjust the sliders and click 'Predict Species' to see the result!")
        
        # Sample predictions
        st.markdown("### 🧪 Try These Sample Measurements")
        
        col1, col2, col3 = st.columns(3)
        
        samples = {
            "Typical Setosa": [5.1, 3.5, 1.4, 0.2],
            "Typical Versicolor": [5.9, 3.0, 4.2, 1.5],
            "Typical Virginica": [6.5, 3.0, 5.5, 2.0]
        }
        
        for i, (name, values) in enumerate(samples.items()):
            with [col1, col2, col3][i]:
                if st.button(f"📊 {name}", key=f"sample_{i}"):
                    species, confidence, proba = predict_species(model, scaler, encoder, *values)
                    st.success(f"**{species}** ({confidence:.1f}% confidence)")
    
    elif page == "📊 Model Performance":
        st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Model evaluation
        loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>{accuracy:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📉 Loss</h3>
            <h2>{loss:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_grade = "Excellent" if accuracy > 0.95 else "Good" if accuracy > 0.90 else "Fair"
            st.markdown(f"""
            <div class="metric-card">
            <h3>⭐ Grade</h3>
            <h2>{performance_grade}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Training history plots (if available)
        if history is not None:
            st.markdown("### 📈 Training History")
            training_fig = create_training_plots(history)
            if training_fig:
                st.pyplot(training_fig)
        
        # Model architecture
        st.markdown("### 🏗️ Model Architecture")
        
        architecture_info = """
        **Neural Network Configuration:**
        - **Input Layer**: 4 neurons (sepal length, sepal width, petal length, petal width)
        - **Hidden Layer 1**: 16 neurons with ReLU activation
        - **Hidden Layer 2**: 8 neurons with ReLU activation
        - **Output Layer**: 3 neurons with Softmax activation
        - **Optimizer**: Adam (learning rate: 0.001)
        - **Loss Function**: Categorical Crossentropy
        - **Training Epochs**: 100
        - **Batch Size**: 8
        """
        
        st.markdown(architecture_info)
        
        # Feature importance (simplified visualization)
        st.markdown("### 📊 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plotly version
            feature_stats = df.groupby('variety')[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].mean()
            fig_features = px.bar(
                feature_stats.T,
                title="Average Feature Values by Species (Plotly)",
                labels={'index': 'Features', 'value': 'Average Value (cm)'},
                barmode='group',
                color_discrete_map={'Setosa': '#2E86AB', 'Versicolor': '#F18F01', 'Virginica': '#C73E1D'}
            )
            fig_features.update_layout(title_font_color="#2E86AB", title_font_size=16)
            st.plotly_chart(fig_features, use_container_width=True)
        
        with col2:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_stats.T.plot(kind='bar', ax=ax, color=['#2E86AB', '#F18F01', '#C73E1D'], alpha=0.8)
            ax.set_title('Average Feature Values by Species (Matplotlib)', fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Average Value (cm)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Species')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif page == "ℹ️ About":
        st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>🎓 Educational Purpose</h3>
        <p>This application demonstrates the implementation of an Artificial Neural Network (ANN) for multi-class classification 
        using the famous Iris dataset from Ronald Fisher's 1936 paper.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔬 Technical Details
        
        **Dataset Information:**
        - **Source**: Fisher's Iris Dataset (1936)
        - **Features**: 4 numerical measurements
        - **Classes**: 3 iris species (Setosa, Versicolor, Virginica)
        - **Size**: 150 samples (50 per class)
        
        **Model Specifications:**
        - **Framework**: TensorFlow/Keras
        - **Architecture**: Sequential Neural Network
        - **Layers**: 2 hidden layers with ReLU activation
        - **Output**: Softmax activation for multi-class probability
        - **Training**: 100 epochs with validation split
        
        **Data Preprocessing:**
        - Feature standardization using StandardScaler
        - Label encoding for categorical targets
        - Stratified train-test split (80/20)
        
        ### 🚀 Key Features
        - **Interactive Prediction**: Real-time species classification
        - **Data Visualization**: Comprehensive exploratory data analysis
        - **Multiple Visualization Libraries**: Plotly, Matplotlib, and Seaborn
        - **Model Performance**: Detailed accuracy and loss metrics
        - **Training History**: Visual representation of model learning
        - **User-Friendly Interface**: Intuitive Streamlit web application
        - **Auto Model Creation**: Creates and trains model if not found
        
        ### 📚 Learning Outcomes
        - Understanding neural network architecture
        - Data preprocessing for machine learning
        - Model evaluation and interpretation
        - Web application development with Streamlit
        - Data visualization with multiple libraries
        
        ### 🛠️ Technologies Used
        - **Python**: Core programming language
        - **TensorFlow/Keras**: Deep learning framework
        - **Streamlit**: Web application framework
        - **Plotly**: Interactive data visualization
        - **Matplotlib**: Static data visualization
        - **Seaborn**: Statistical data visualization
        - **Scikit-learn**: Machine learning utilities
        - **Pandas/NumPy**: Data manipulation and analysis
        
        ---
        
        **📧 Contact**: For questions or suggestions about this project
        
        **🔗 Repository**: [GitHub Link] (Add your repository link here)
        """)

if __name__ == "__main__":
    main()