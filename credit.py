import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Worthiness Prediction", 
    layout="wide",
    page_icon="💳",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .prediction-success {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .prediction-danger {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header with improved styling
st.markdown('<h1 class="main-header">💳 Customer Creditworthiness Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Analyze credit data and predict customer creditworthiness using machine learning")

# Sidebar for navigation with improved styling
st.sidebar.markdown("### 📊 Navigation")
st.sidebar.markdown("---")
options = ["🏠 Data Overview", "📈 Feature Analysis", "🎯 Model Performance", "🔮 Prediction Tool"]
selection = st.sidebar.radio("Choose a section:", options, index=0)

# Add sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This dashboard helps analyze customer creditworthiness using machine learning. "
    "Navigate through different sections to explore data, analyze features, evaluate model performance, and make predictions."
)

# Enhanced data loading with better error handling
@st.cache_data
def load_data():
    """Load and validate the credit risk dataset"""
    file_paths = [
        '/Users/daniyalrosli/Customer-Creditworthiness-Prediction-Using-Machine-Learning/credit_risk_dataset.csv',
        'credit_risk_dataset.csv',
        'data/credit_risk_dataset.csv'
    ]
    
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            # Basic data validation
            if df.empty:
                continue
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"Error reading file {path}: {str(e)}")
            continue
    
    # Generate sample data if no file found
    st.warning("⚠️ Data file not found. Using sample data for demonstration.")
    return generate_sample_data()

def generate_sample_data():
    """Generate sample credit data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'person_age': np.random.randint(18, 80, n_samples),
        'person_income': np.random.lognormal(10, 1, n_samples).astype(int),
        'person_emp_length': np.random.randint(0, 40, n_samples),
        'loan_amnt': np.random.randint(1000, 50000, n_samples),
        'loan_int_rate': np.random.uniform(5, 25, n_samples),
        'person_home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n_samples),
        'loan_intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], n_samples),
        'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
        'cb_person_default_on_file': np.random.choice(['Y', 'N'], n_samples),
        'cb_person_cred_hist_length': np.random.randint(1, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Create target variable with some logic
    df['loan_status'] = ((df['person_income'] > 50000) & 
                        (df['loan_int_rate'] < 15) & 
                        (df['cb_person_default_on_file'] == 'N')).astype(int)
    
    return df

# Enhanced model loading
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_paths = [
        'models/credit_model.pkl',
        'credit_model.pkl'
    ]
    
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            continue
        except Exception as e:
            st.error(f"Error loading model {path}: {str(e)}")
            continue
    
    # Train a simple model if no pre-trained model found
    st.warning("⚠️ Pre-trained model not found. Training a new model with available data.")
    return train_simple_model()

def train_simple_model():
    """Train a simple model with the available data"""
    df = load_data()
    if df is None:
        return None
    
    try:
        # Prepare features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = 'loan_status' if 'loan_status' in df.columns else df.columns[-1]
        
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        X = df[numeric_features].fillna(df[numeric_features].median())
        y = df[target_col]
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# Main content based on selection
if selection == "🏠 Data Overview":
    st.header("📊 Data Overview")
    
    df = load_data()
    if df is not None:
        # Display data info in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", f"{df.shape[0]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", df.shape[1])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("Missing Data", f"{missing_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data sample with search and filter
        st.subheader("🔍 Data Explorer")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search in data:", placeholder="Enter search term...")
        with col2:
            show_rows = st.selectbox("Rows to show:", [10, 25, 50, 100], index=1)
        
        # Filter data based on search
        display_df = df.head(show_rows)
        if search_term:
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = df[mask].head(show_rows)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download data option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="credit_data.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Enhanced statistical summary
        st.subheader("📈 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
        
        with tab1:
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric features found in the dataset.")
        
        with tab2:
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    st.write(f"**{col}**")
                    value_counts = df[col].value_counts()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                   title=f"Distribution of {col}")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(value_counts.head(10))
                    st.markdown("---")
            else:
                st.info("No categorical features found in the dataset.")
        
        # Target distribution (if exists)
        target_cols = ['loan_status', 'credit_worthy', 'default', 'target']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            st.subheader("🎯 Target Variable Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_counts = df[target_col].value_counts()
                fig = px.pie(values=target_counts.values, names=target_counts.index,
                           title=f"Distribution of {target_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                target_pct = df[target_col].value_counts(normalize=True) * 100
                st.markdown("**Class Distribution:**")
                for idx, (class_name, percentage) in enumerate(target_pct.items()):
                    st.write(f"• Class {class_name}: {percentage:.2f}%")
                
                # Class balance indicator
                balance_ratio = target_pct.min() / target_pct.max()
                if balance_ratio > 0.8:
                    st.success("✅ Well-balanced dataset")
                elif balance_ratio > 0.5:
                    st.warning("⚠️ Slightly imbalanced dataset")
                else:
                    st.error("❌ Highly imbalanced dataset")

elif selection == "📈 Feature Analysis":
    st.header("📈 Feature Analysis")
    
    df = load_data()
    if df is not None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) > 1:
            # Interactive correlation heatmap
            st.subheader("🔥 Correlation Analysis")
            
            col1, col2 = st.columns([3, 1])
            with col2:
                corr_method = st.selectbox("Correlation Method:", ["pearson", "spearman", "kendall"])
                min_corr = st.slider("Minimum Correlation to Show:", 0.0, 1.0, 0.0, 0.1)
            
            with col1:
                correlation = df[numeric_columns].corr(method=corr_method)
                
                # Filter correlations
                mask = np.abs(correlation) >= min_corr
                filtered_corr = correlation.where(mask)
                
                fig = px.imshow(filtered_corr, 
                              title=f"{corr_method.title()} Correlation Matrix",
                              color_continuous_scale="RdBu_r",
                              aspect="auto")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # High correlation pairs
            corr_pairs = []
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    corr_val = correlation.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        corr_pairs.append({
                            'Feature 1': correlation.columns[i],
                            'Feature 2': correlation.columns[j],
                            'Correlation': corr_val
                        })
            
            if corr_pairs:
                st.subheader("🚨 High Correlation Pairs (|r| > 0.7)")
                corr_df = pd.DataFrame(corr_pairs)
                st.dataframe(corr_df.sort_values('Correlation', key=abs, ascending=False))
        
        st.markdown("---")
        
        # Enhanced feature distributions
        st.subheader("📊 Feature Distribution Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            feature = st.selectbox("Select Feature for Analysis:", 
                                 numeric_columns + categorical_columns)
        with col2:
            plot_type = st.selectbox("Plot Type:", 
                                   ["Histogram", "Box Plot", "Violin Plot"] if feature in numeric_columns 
                                   else ["Bar Chart", "Pie Chart"])
        with col3:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if feature in numeric_columns:
                if plot_type == "Histogram":
                    fig = px.histogram(df, x=feature, nbins=30, 
                                     title=f"Distribution of {feature}")
                elif plot_type == "Box Plot":
                    fig = px.box(df, y=feature, title=f"Box Plot of {feature}")
                else:  # Violin Plot
                    fig = px.violin(df, y=feature, title=f"Violin Plot of {feature}")
            else:
                value_counts = df[feature].value_counts()
                if plot_type == "Bar Chart":
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f"Distribution of {feature}")
                else:  # Pie Chart
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                               title=f"Distribution of {feature}")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if show_stats:
                st.markdown(f"**Statistics for {feature}:**")
                
                if feature in numeric_columns:
                    stats = df[feature].describe()
                    for stat, value in stats.items():
                        st.write(f"• {stat.title()}: {value:.2f}")
                    
                    # Additional statistics
                    st.write(f"• Skewness: {df[feature].skew():.2f}")
                    st.write(f"• Kurtosis: {df[feature].kurtosis():.2f}")
                    st.write(f"• Missing Values: {df[feature].isnull().sum()}")
                    
                else:
                    value_counts = df[feature].value_counts()
                    st.write(f"• Unique Values: {df[feature].nunique()}")
                    st.write(f"• Most Common: {value_counts.index[0]} ({value_counts.iloc[0]})")
                    st.write(f"• Missing Values: {df[feature].isnull().sum()}")
                    
                    st.markdown("**Top Categories:**")
                    for idx, (cat, count) in enumerate(value_counts.head(5).items()):
                        pct = count / len(df) * 100
                        st.write(f"{idx+1}. {cat}: {count} ({pct:.1f}%)")
        
        # Target relationship analysis
        target_cols = ['loan_status', 'credit_worthy', 'default', 'target']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col and feature != target_col:
            st.markdown("---")
            st.subheader(f"🎯 Relationship with Target ({target_col})")
            
            if feature in numeric_columns:
                fig = px.box(df, x=target_col, y=feature, 
                           title=f"{feature} by {target_col}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Cross-tabulation for categorical features
                crosstab = pd.crosstab(df[feature], df[target_col], normalize='index') * 100
                fig = px.bar(crosstab, title=f"{feature} by {target_col} (%)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif selection == "🎯 Model Performance":
    st.header("🎯 Model Performance Analysis")
    
    df = load_data()
    model = load_model()
    
    if df is not None and model is not None:
        # Find target column
        target_cols = ['loan_status', 'credit_worthy', 'default', 'target']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            try:
                # Prepare data
                numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_col in numeric_features:
                    numeric_features.remove(target_col)
                
                X = df[numeric_features].fillna(df[numeric_features].median())
                y = df[target_col]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Display metrics with improved styling
                st.subheader("📊 Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{acc:.3f}", f"{acc*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Precision", f"{prec:.3f}", f"{prec*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Recall", f"{rec:.3f}", f"{rec*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("F1 Score", f"{f1:.3f}", f"{f1*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                # Enhanced Confusion Matrix
                with col1:
                    st.subheader("🔍 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(cm, 
                                  text_auto=True, 
                                  aspect="auto",
                                  title="Confusion Matrix",
                                  labels=dict(x="Predicted", y="Actual"),
                                  color_continuous_scale="Blues")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve (if probabilities available)
                with col2:
                    if y_prob is not None:
                        st.subheader("📈 ROC Curve")
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                               name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                               line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                               mode='lines',
                                               name='Random Classifier',
                                               line=dict(color='red', dash='dash')))
                        fig.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ROC curve not available for this model type.")
                
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    st.markdown("---")
                    st.subheader("🎯 Feature Importance Analysis")
                    
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        top_n = st.slider("Number of top features to show:", 5, len(X.columns), 10)
                        
                        fig = px.bar(feature_importance_df.head(top_n), 
                                   x='Importance', y='Feature',
                                   orientation='h',
                                   title=f'Top {top_n} Feature Importances')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Feature Importance Ranking:**")
                        for idx, row in feature_importance_df.head(10).iterrows():
                            st.write(f"{row.name + 1}. {row['Feature']}: {row['Importance']:.4f}")
                        
                        # Download feature importance
                        csv = feature_importance_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Feature Importance",
                            data=csv,
                            file_name="feature_importance.csv",
                            mime="text/csv"
                        )
                
                # Model interpretation
                st.markdown("---")
                st.subheader("🧠 Model Interpretation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Model Strengths:**")
                    if acc > 0.8:
                        st.success("✅ High accuracy performance")
                    if prec > 0.8:
                        st.success("✅ Good precision - low false positives")
                    if rec > 0.8:
                        st.success("✅ Good recall - captures most positive cases")
                    if f1 > 0.8:
                        st.success("✅ Well-balanced F1 score")
                
                with col2:
                    st.markdown("**Areas for Improvement:**")
                    if acc < 0.7:
                        st.warning("⚠️ Consider improving overall accuracy")
                    if prec < 0.7:
                        st.warning("⚠️ High false positive rate")
                    if rec < 0.7:
                        st.warning("⚠️ Missing many positive cases")
                    if abs(prec - rec) > 0.1:
                        st.warning("⚠️ Imbalanced precision/recall trade-off")
                
            except Exception as e:
                st.error(f"Error in model evaluation: {str(e)}")
                st.info("Please check your data format and model compatibility.")
        else:
            st.warning("No target column found. Please ensure your dataset has a target variable.")
    else:
        st.error("Unable to load data or model for performance analysis.")

elif selection == "🔮 Prediction Tool":
    st.header("🔮 Credit Worthiness Prediction Tool")
    
    model = load_model()
    df = load_data()
    
    if model is not None and df is not None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Instructions:** Fill in the customer information below to get a creditworthiness prediction. All fields are required for accurate prediction.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get numeric columns for input
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target columns
        target_cols = ['loan_status', 'credit_worthy', 'default', 'target']
        for col in target_cols:
            if col in numeric_columns:
                numeric_columns.remove(col)
        
        if len(numeric_columns) > 0:
            # Create input form with better organization
            st.subheader("📝 Customer Information Form")
            
            # Organize inputs in columns
            num_cols = min(3, len(numeric_columns))
            cols = st.columns(num_cols)
            
            input_data = {}
            col_idx = 0
            
            for feature in numeric_columns:
                with cols[col_idx % num_cols]:
                    # Get reasonable defaults based on data
                    feature_data = df[feature].dropna()
                    min_val = float(feature_data.min())
                    max_val = float(feature_data.max())
                    default_val = float(feature_data.median())
                    
                    # Special handling for common features
                    if 'age' in feature.lower():
                        input_data[feature] = st.number_input(
                            f"Age", 
                            min_value=18, max_value=100, value=35,
                            help="Customer's age in years"
                        )
                    elif 'income' in feature.lower():
                        input_data[feature] = st.number_input(
                            f"Annual Income", 
                            min_value=0, value=50000, step=1000,
                            help="Annual income in dollars"
                        )
                    elif 'loan' in feature.lower() and 'amount' in feature.lower():
                        input_data[feature] = st.number_input(
                            f"Loan Amount", 
                            min_value=0, value=int(default_val), step=100,
                            help="Requested loan amount"
                        )
                    elif 'rate' in feature.lower():
                        input_data[feature] = st.slider(
                            f"Interest Rate (%)", 
                            min_value=0.0, max_value=30.0, value=float(default_val),
                            help="Interest rate percentage"
                        )
                    elif 'employment' in feature.lower() or 'emp' in feature.lower():
                        input_data[feature] = st.number_input(
                            f"Employment Length (years)", 
                            min_value=0, value=int(default_val),
                            help="Years of employment"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}", 
                            min_value=min_val, max_value=max_val, value=default_val,
                            help=f"Enter value for {feature.replace('_', ' ')}"
                        )
                
                col_idx += 1
            
            st.markdown("---")
            
            # Prediction section
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                predict_button = st.button("🔮 Predict Credit Worthiness", 
                                         type="primary", 
                                         use_container_width=True)
            
            if predict_button:
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Handle missing features that model expects
                    model_features = getattr(model, 'feature_names_in_', numeric_columns)
                    for feature in model_features:
                        if feature not in input_df.columns:
                            # Use median from training data if available
                            if feature in df.columns:
                                input_df[feature] = df[feature].median()
                            else:
                                input_df[feature] = 0
                    
                    # Ensure correct column order
                    input_df = input_df.reindex(columns=model_features, fill_value=0)
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Get prediction probability if available
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(input_df)[0]
                        confidence = max(probabilities)
                        prob_positive = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                    else:
                        confidence = 0.8  # Default confidence
                        prob_positive = float(prediction)
                    
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Display prediction with enhanced styling
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-success">'
                            f'<h3>✅ Credit Approved</h3>'
                            f'<p>The customer is likely <strong>credit worthy</strong></p>'
                            f'<p>Confidence: <strong>{confidence:.1%}</strong></p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-danger">'
                            f'<h3>❌ Credit Declined</h3>'
                            f'<p>The customer is likely <strong>not credit worthy</strong></p>'
                            f'<p>Confidence: <strong>{confidence:.1%}</strong></p>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                    
                    # Probability visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Probability Breakdown")
                        
                        prob_data = pd.DataFrame({
                            'Outcome': ['Not Credit Worthy', 'Credit Worthy'],
                            'Probability': [1 - prob_positive, prob_positive]
                        })
                        
                        fig = px.bar(prob_data, x='Outcome', y='Probability', 
                                   title='Prediction Probabilities',
                                   color='Probability',
                                   color_continuous_scale=['red', 'green'])
                        fig.update_layout(height=400, showlegend=False)
                        fig.update_traces(texttemplate='%{y:.1%}', textposition='auto')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎚️ Risk Assessment")
                        
                        # Risk level based on probability
                        if prob_positive >= 0.8:
                            risk_level = "Low Risk"
                            risk_color = "green"
                            risk_icon = "✅"
                        elif prob_positive >= 0.6:
                            risk_level = "Medium Risk"
                            risk_color = "orange"  
                            risk_icon = "⚠️"
                        else:
                            risk_level = "High Risk"
                            risk_color = "red"
                            risk_icon = "❌"
                        
                        st.markdown(f"**Risk Level:** {risk_icon} {risk_level}")
                        st.markdown(f"**Approval Probability:** {prob_positive:.1%}")
                        
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prob_positive * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Credit Score"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgray"},
                                    {'range': [40, 70], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature contribution analysis (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("---")
                        st.subheader("🔍 Factor Analysis")
                        
                        # Get feature importances
                        feature_importance = model.feature_importances_
                        feature_names = model_features
                        
                        # Create contribution analysis
                        contribution_data = []
                        for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                            if feature in input_data:
                                value = input_data[feature]
                                contribution_data.append({
                                    'Feature': feature.replace('_', ' ').title(),
                                    'Value': value,
                                    'Importance': importance,
                                    'Contribution': importance * value
                                })
                        
                        contrib_df = pd.DataFrame(contribution_data).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Key Contributing Factors:**")
                            top_factors = contrib_df.head(5)
                            
                            fig = px.bar(top_factors, x='Importance', y='Feature',
                                       orientation='h', title='Top Contributing Features')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Your Input Summary:**")
                            for _, row in top_factors.iterrows():
                                st.write(f"• **{row['Feature']}**: {row['Value']}")
                            
                            # Recommendations
                            st.markdown("**💡 Recommendations:**")
                            if prob_positive < 0.6:
                                st.write("• Consider improving credit history")
                                st.write("• Increase income or reduce debt")
                                st.write("• Consider a co-signer")
                            else:
                                st.write("• Strong credit profile")
                                st.write("• Good approval chances")
                                st.write("• Consider negotiating better rates")
                    
                    # Export prediction report
                    st.markdown("---")
                    
                    # Create report data
                    report_data = {
                        'Customer Information': input_data,
                        'Prediction': 'Credit Worthy' if prediction == 1 else 'Not Credit Worthy',
                        'Probability': f"{prob_positive:.1%}",
                        'Risk Level': risk_level,
                        'Confidence': f"{confidence:.1%}",
                        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Convert to DataFrame for export
                    report_df = pd.DataFrame([{
                        'Feature': k,
                        'Value': v
                    } for k, v in input_data.items()])
                    
                    report_df = pd.concat([
                        report_df,
                        pd.DataFrame([
                            {'Feature': 'Prediction', 'Value': report_data['Prediction']},
                            {'Feature': 'Probability', 'Value': report_data['Probability']},
                            {'Feature': 'Risk Level', 'Value': report_data['Risk Level']},
                            {'Feature': 'Confidence', 'Value': report_data['Confidence']},
                            {'Feature': 'Timestamp', 'Value': report_data['Timestamp']}
                        ])
                    ])
                    
                    csv_report = report_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=csv_report,
                        file_name=f"credit_prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("Please check your input values and try again.")
                    
                    # Debug information for developers
                    with st.expander("🔧 Debug Information"):
                        st.write("**Input Data:**", input_data)
                        st.write("**Model Features:**", getattr(model, 'feature_names_in_', 'Not available'))
                        st.write("**Error Details:**", str(e))
        
        else:
            st.warning("⚠️ No numeric features found in the dataset for prediction.")
            st.info("Please ensure your dataset contains numeric features for model input.")
    
    else:
        st.error("❌ Unable to load model or data for predictions.")
        st.info("Please check your data and model files.")

# Enhanced Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏦 Credit Dashboard")
    st.markdown("Advanced ML-powered credit analysis")

with col2:
    st.markdown("### 📊 Features")
    st.markdown("• Interactive data exploration")
    st.markdown("• Real-time predictions") 
    st.markdown("• Performance analytics")
    st.markdown("• Export capabilities")

with col3:
    st.markdown("### 🔧 Built With")
    st.markdown("• Streamlit")
    st.markdown("• Scikit-learn")
    st.markdown("• Plotly")
    st.markdown("• Pandas")

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9em;">'
    '💳 Customer Creditworthiness Prediction Dashboard | '
    'Built with ❤️ using Streamlit'
    '</div>', 
    unsafe_allow_html=True
)