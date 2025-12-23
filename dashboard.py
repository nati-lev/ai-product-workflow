# -*- coding: utf-8 -*-
"""
AI Product Workflow Dashboard
Streamlit app for visualizing results
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="AI Product Workflow Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_validation_report():
    """Load validation report"""
    path = Path('artifacts/analyst/validation_report.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_evaluation_report():
    """Load evaluation report"""
    path = Path('artifacts/scientist/evaluation_report.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_clean_data():
    """Load clean dataset"""
    path = Path('artifacts/analyst/clean_data.csv')
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_insights():
    """Load insights"""
    path = Path('artifacts/analyst/insights.md')
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">AI Product Workflow Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("### End-to-End ML Pipeline Results")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Data Analysis", "Model Performance", "Documentation"]
    )
    
    # Load data
    validation_report = load_validation_report()
    eval_report = load_evaluation_report()
    clean_data = load_clean_data()
    insights = load_insights()
    
    # Check if data exists
    if not eval_report:
        st.error("Evaluation report not found! Please run the pipeline first.")
        st.info("Run: python create_summary.py")
        return
    
    # Page routing
    if page == "Overview":
        show_overview(validation_report, eval_report, clean_data)
    elif page == "Data Analysis":
        show_data_analysis(validation_report, clean_data, insights)
    elif page == "Model Performance":
        show_model_performance(eval_report)
    elif page == "Documentation":
        show_documentation()

def show_overview(validation_report, eval_report, clean_data):
    """Overview page"""
    st.header("Project Overview")
    
    # Status badges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="success-badge">Data Analysis: COMPLETE</div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="success-badge">Model Training: COMPLETE</div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="success-badge">Status: READY</div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    st.subheader("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if validation_report:
            st.metric("Total Rows", "{:,}".format(validation_report['total_rows']))
        else:
            st.metric("Total Rows", "N/A")
    
    with col2:
        if validation_report:
            st.metric("Total Columns", validation_report['total_columns'])
        else:
            st.metric("Total Columns", "N/A")
    
    with col3:
        if eval_report:
            st.metric("Features Used", eval_report['features_count'])
        else:
            st.metric("Features Used", "N/A")
    
    with col4:
        if eval_report:
            best_model = eval_report['best_model']
            accuracy = eval_report['all_results'][best_model]['accuracy']
            st.metric("Model Accuracy", "{:.1%}".format(accuracy))
        else:
            st.metric("Model Accuracy", "N/A")
    
    st.markdown("---")
    
    # Pipeline flow
    st.subheader("Pipeline Flow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Phase 1: Data Analysis")
        st.markdown("""
        1. Data Validation
        2. Data Cleaning
        3. Exploratory Analysis
        4. Schema Design
        """)
        
        if validation_report:
            st.success("Status: Complete")
            st.write("Missing Values:", validation_report.get('total_missing', 'N/A'))
            st.write("Duplicates:", validation_report.get('duplicates', 'N/A'))
    
    with col2:
        st.markdown("#### Phase 2: Model Development")
        st.markdown("""
        1. Feature Engineering
        2. Model Training
        3. Model Evaluation
        4. Documentation
        """)
        
        if eval_report:
            st.success("Status: Complete")
            st.write("Models Trained:", len(eval_report['all_results']))
            st.write("Best Model:", eval_report['best_model'].replace('_', ' ').title())

def show_data_analysis(validation_report, clean_data, insights):
    """Data analysis page"""
    st.header("Data Analysis Results")
    
    tab1, tab2, tab3 = st.tabs(["Validation", "Dataset Preview", "Insights"])
    
    with tab1:
        st.subheader("Data Validation")
        
        if validation_report:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", "{:,}".format(validation_report['total_rows']))
            with col2:
                st.metric("Total Columns", validation_report['total_columns'])
            with col3:
                st.metric("Duplicates", validation_report.get('duplicates', 0))
            
            st.markdown("---")
            st.subheader("Summary")
            st.info(validation_report.get('summary', 'No summary available'))
            
            # Missing values chart
            if 'columns' in validation_report:
                missing_data = []
                for col, info in validation_report['columns'].items():
                    if info['missing_count'] > 0:
                        missing_data.append({
                            'Column': col,
                            'Missing Count': info['missing_count'],
                            'Missing %': info['missing_percentage']
                        })
                
                if missing_data:
                    st.subheader("Missing Values")
                    df_missing = pd.DataFrame(missing_data)
                    
                    fig = px.bar(df_missing, x='Column', y='Missing Count',
                               title='Missing Values by Column',
                               color='Missing %',
                               color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Validation report not found")
    
    with tab2:
        st.subheader("Dataset Preview")
        
        if clean_data is not None:
            st.write("Shape:", clean_data.shape)
            st.dataframe(clean_data.head(100), use_container_width=True)
            
            # Data types
            st.subheader("Column Types")
            dtype_df = pd.DataFrame({
                'Column': clean_data.columns,
                'Type': clean_data.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
        else:
            st.warning("Clean data not found")
    
    with tab3:
        st.subheader("EDA Insights")
        
        if insights:
            st.markdown(insights)
        else:
            st.warning("Insights not found")
            
        # Show plots if they exist
        plots_dir = Path('artifacts/analyst')
        if plots_dir.exists():
            plot_files = list(plots_dir.glob('*.png'))
            
            if plot_files:
                st.subheader("Visualizations")
                
                # Show plots in grid
                cols = st.columns(2)
                for idx, plot_file in enumerate(plot_files[:6]):  # Show first 6
                    with cols[idx % 2]:
                        st.image(str(plot_file), caption=plot_file.name, 
                               use_container_width=True)

def show_model_performance(eval_report):
    """Model performance page"""
    st.header("Model Performance")
    
    if not eval_report:
        st.warning("Evaluation report not found")
        return
    
    best_model = eval_report['best_model']
    best_results = eval_report['all_results'][best_model]
    
    # Best model metrics
    st.subheader("Best Model: {}".format(best_model.replace('_', ' ').title()))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "{:.2%}".format(best_results['accuracy']))
    with col2:
        st.metric("Precision", "{:.2%}".format(best_results['precision']))
    with col3:
        st.metric("Recall", "{:.2%}".format(best_results['recall']))
    with col4:
        st.metric("F1 Score", "{:.2%}".format(best_results['f1_score']))
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    comparison_data = []
    for model_name, results in eval_report['all_results'].items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1_score'],
            'Best': 'Yes' if model_name == best_model else 'No'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Bar chart
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_comparison['Model'],
            y=df_comparison[metric],
            text=['{:.1%}'.format(v) for v in df_comparison[metric]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(df_comparison, use_container_width=True)
    
    # Confusion matrix
    st.markdown("---")
    st.subheader("Confusion Matrix (Best Model)")
    
    if 'confusion_matrix' in best_results:
        cm = best_results['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Training info
    st.markdown("---")
    st.subheader("Training Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Samples:**", "{:,}".format(eval_report['training_samples']))
        st.write("**Test Samples:**", "{:,}".format(eval_report['test_samples']))
    
    with col2:
        st.write("**Features Count:**", eval_report['features_count'])
        st.write("**Target Column:**", eval_report['target_column'])

def show_documentation():
    """Documentation page"""
    st.header("Project Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Model Card", "Artifacts", "Usage Guide"])
    
    with tab1:
        st.subheader("Model Card")
        
        model_card_path = Path('artifacts/scientist/model_card.md')
        if model_card_path.exists():
            with open(model_card_path, 'r', encoding='utf-8') as f:
                model_card = f.read()
            st.markdown(model_card)
        else:
            st.warning("Model card not found")
    
    with tab2:
        st.subheader("Generated Artifacts")
        
        st.markdown("#### Data Analysis Artifacts")
        st.code("""
artifacts/analyst/
├── validation_report.json
├── clean_data.csv
├── insights.md
├── dataset_contract.json
└── *.png (plots)
        """)
        
        st.markdown("#### Model Artifacts")
        st.code("""
artifacts/scientist/
├── features.csv
├── model.pkl
├── evaluation_report.json
└── model_card.md
        """)
        
        # File sizes
        st.markdown("#### File Information")
        
        files_info = []
        for artifact_dir in ['artifacts/analyst', 'artifacts/scientist']:
            path = Path(artifact_dir)
            if path.exists():
                for file in path.iterdir():
                    if file.is_file():
                        size = file.stat().st_size
                        files_info.append({
                            'File': file.name,
                            'Location': artifact_dir,
                            'Size (KB)': round(size / 1024, 2)
                        })
        
        if files_info:
            df_files = pd.DataFrame(files_info)
            st.dataframe(df_files, use_container_width=True)
    
    with tab3:
        st.subheader("Usage Guide")
        
        st.markdown("""
        ### How to Use This Project
        
        #### 1. Review Results
        - Navigate through the dashboard tabs
        - Check model performance metrics
        - Review data analysis insights
        
        #### 2. Use the Model
```python
        import joblib
        import pandas as pd
        
        # Load the model
        model = joblib.load('artifacts/scientist/model.pkl')
        
        # Load features (for reference)
        features = pd.read_csv('artifacts/scientist/features.csv')
        
        # Make predictions
        predictions = model.predict(new_data)
```
        
        #### 3. Deploy to Production
        - Review model_card.md for deployment guidelines
        - Set up monitoring for model performance
        - Schedule retraining every 3-6 months
        
        #### 4. Next Steps
        - Test with new data
        - Create API endpoint
        - Set up CI/CD pipeline
        - Implement monitoring dashboard
        
        ### Contact
        For questions or issues, refer to the project documentation.
        """)

if __name__ == "__main__":
    main()