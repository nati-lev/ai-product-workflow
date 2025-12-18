# תבניות קוד מוכנות לשימוש

## 🎯 Crew 1: Data Analyst - Template מלא

### agents.py
```python
from crewai import Agent
from crewai_tools import FileReadTool, CSVSearchTool
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

# Define tools
file_reader = FileReadTool()
csv_tool = CSVSearchTool()

# Agent 1: Data Validator
data_validator = Agent(
    role='Data Quality Validator',
    goal='Ensure data quality, identify missing values, outliers, and data integrity issues',
    backstory='''You are an expert in data quality assessment with years of experience 
    in validating datasets for machine learning projects. You have a keen eye for spotting 
    data anomalies and ensuring data meets quality standards.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[file_reader, csv_tool]
)

# Agent 2: Data Cleaner
data_cleaner = Agent(
    role='Data Cleaning Specialist',
    goal='Clean and preprocess data by handling missing values, removing duplicates, and standardizing formats',
    backstory='''You are a meticulous data engineer who specializes in data cleaning 
    and preprocessing. You follow best practices to ensure data is clean, consistent, 
    and ready for analysis.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[csv_tool]
)

# Agent 3: EDA Analyst
eda_analyst = Agent(
    role='Exploratory Data Analyst',
    goal='Perform comprehensive exploratory data analysis and create insightful visualizations',
    backstory='''You are a data scientist specialized in exploratory data analysis. 
    You excel at uncovering patterns, relationships, and insights through statistical 
    analysis and compelling visualizations.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[csv_tool]
)

# Agent 4: Schema Designer
schema_designer = Agent(
    role='Data Contract Designer',
    goal='Create comprehensive dataset contracts that define schema, constraints, and validation rules',
    backstory='''You are a data architect who specializes in creating robust data contracts. 
    You ensure that downstream systems have clear expectations about data structure and quality.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)
```

### tasks.py
```python
from crewai import Task
from .agents import data_validator, data_cleaner, eda_analyst, schema_designer
import os

# Get data path
DATA_PATH = os.getenv('RAW_DATA_PATH', 'data/raw/dataset.csv')
OUTPUT_DIR = 'artifacts/analyst'

# Task 1: Data Validation
validate_data_task = Task(
    description=f'''
    Validate the dataset at {DATA_PATH}:
    1. Check for missing values (count and percentage per column)
    2. Identify duplicate rows
    3. Detect outliers using IQR method
    4. Verify data types are appropriate
    5. Check for inconsistent formatting
    6. Generate a validation report with findings
    
    Save report to {OUTPUT_DIR}/validation_report.json
    ''',
    agent=data_validator,
    expected_output='JSON file with validation results including missing values, duplicates, and outliers'
)

# Task 2: Data Cleaning
clean_data_task = Task(
    description=f'''
    Clean the validated dataset:
    1. Handle missing values:
       - Numerical: Use median/mean imputation based on distribution
       - Categorical: Use mode or create 'Unknown' category
    2. Remove duplicate rows
    3. Handle outliers (cap or remove based on severity)
    4. Standardize formats (dates, strings, etc.)
    5. Ensure consistent naming conventions
    
    Save cleaned dataset to {OUTPUT_DIR}/clean_data.csv
    Log all cleaning operations performed
    ''',
    agent=data_cleaner,
    expected_output='Clean CSV file and cleaning log',
    context=[validate_data_task]
)

# Task 3: Exploratory Data Analysis
eda_task = Task(
    description=f'''
    Perform comprehensive EDA on the cleaned dataset:
    1. Summary statistics for all columns
    2. Distribution analysis (histograms, box plots)
    3. Correlation analysis (heatmap)
    4. Categorical variable analysis (bar charts)
    5. Identify key patterns and relationships
    6. Generate business insights
    
    Create an HTML report with all visualizations
    Save to {OUTPUT_DIR}/eda_report.html
    Write key insights to {OUTPUT_DIR}/insights.md
    ''',
    agent=eda_analyst,
    expected_output='HTML report with visualizations and Markdown insights document',
    context=[clean_data_task]
)

# Task 4: Create Dataset Contract
create_contract_task = Task(
    description=f'''
    Create a comprehensive dataset contract:
    1. Define schema (column names, types, nullable)
    2. Specify value ranges and constraints
    3. Document assumptions about the data
    4. List validation rules
    5. Define data quality metrics
    
    Save to {OUTPUT_DIR}/dataset_contract.json
    
    JSON structure:
    {{
        "schema_version": "1.0",
        "dataset_name": "cleaned_dataset",
        "columns": {{
            "column_name": {{
                "type": "data_type",
                "nullable": true/false,
                "range": [min, max],
                "allowed_values": [],
                "description": ""
            }}
        }},
        "constraints": [],
        "assumptions": [],
        "quality_metrics": {{}}
    }}
    ''',
    agent=schema_designer,
    expected_output='JSON file defining the dataset contract',
    context=[clean_data_task, eda_task]
)
```

### crew.py
```python
from crewai import Crew, Process
from .agents import data_validator, data_cleaner, eda_analyst, schema_designer
from .tasks import validate_data_task, clean_data_task, eda_task, create_contract_task

class AnalystCrew:
    def __init__(self):
        self.crew = Crew(
            agents=[
                data_validator,
                data_cleaner,
                eda_analyst,
                schema_designer
            ],
            tasks=[
                validate_data_task,
                clean_data_task,
                eda_task,
                create_contract_task
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def kickoff(self):
        """Execute the analyst crew"""
        print("🚀 Starting Data Analyst Crew...")
        result = self.crew.kickoff()
        print("✅ Data Analyst Crew completed!")
        return result

if __name__ == "__main__":
    crew = AnalystCrew()
    crew.kickoff()
```

---

## 🔬 Crew 2: Data Scientist - Template מלא

### agents.py
```python
from crewai import Agent
from crewai_tools import FileReadTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
file_reader = FileReadTool()

# Agent 1: Contract Validator
contract_validator = Agent(
    role='Dataset Contract Validator',
    goal='Validate that cleaned data matches the dataset contract specifications',
    backstory='''You are a quality assurance expert who ensures data contracts 
    are followed precisely. You meticulously check every constraint and validation rule.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[file_reader]
)

# Agent 2: Feature Engineer
feature_engineer = Agent(
    role='Feature Engineering Specialist',
    goal='Create meaningful features that improve model performance',
    backstory='''You are a data scientist with deep expertise in feature engineering. 
    You know how to transform raw data into powerful predictive features using domain 
    knowledge and statistical techniques.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 3: Model Trainer
model_trainer = Agent(
    role='Machine Learning Engineer',
    goal='Train and optimize machine learning models for maximum performance',
    backstory='''You are an ML engineer with extensive experience in training various 
    models. You understand hyperparameter tuning, cross-validation, and best practices 
    for model training.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 4: Model Evaluator
model_evaluator = Agent(
    role='Model Evaluation Expert',
    goal='Evaluate models comprehensively and compare their performance',
    backstory='''You are an expert in model evaluation and selection. You analyze 
    multiple metrics, create comparison reports, and recommend the best model based 
    on business requirements.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 5: Documentation Specialist
doc_specialist = Agent(
    role='ML Documentation Specialist',
    goal='Create comprehensive model cards that document everything about the model',
    backstory='''You are specialized in ML documentation and responsible AI practices. 
    You create detailed model cards that ensure transparency and accountability.''',
    verbose=True,
    allow_delegation=False,
    llm=llm
)
```

### tasks.py
```python
from crewai import Task
from .agents import (contract_validator, feature_engineer, 
                     model_trainer, model_evaluator, doc_specialist)

OUTPUT_DIR = 'artifacts/scientist'

# Task 1: Validate Contract
validate_contract_task = Task(
    description='''
    Validate that the cleaned dataset matches the dataset contract:
    1. Load clean_data.csv and dataset_contract.json
    2. Verify all columns exist with correct types
    3. Check value ranges match constraints
    4. Validate row count meets minimums
    5. Ensure all validation rules pass
    
    Save validation results to {OUTPUT_DIR}/validation_status.json
    If validation fails, provide detailed error report
    ''',
    agent=contract_validator,
    expected_output='JSON validation status with pass/fail for each rule'
)

# Task 2: Feature Engineering
feature_engineering_task = Task(
    description=f'''
    Engineer features for predictive modeling:
    1. Create interaction features (e.g., A*B, A/B)
    2. Encode categorical variables (one-hot or label encoding)
    3. Scale numerical features (StandardScaler)
    4. Create polynomial features if beneficial
    5. Handle date/time features (extract day, month, hour)
    6. Log transformations for skewed distributions
    
    Document all transformations performed
    Save feature matrix to {OUTPUT_DIR}/features.csv
    Save transformation pipeline to {OUTPUT_DIR}/feature_pipeline.pkl
    Create log at {OUTPUT_DIR}/feature_engineering_log.md
    ''',
    agent=feature_engineer,
    expected_output='Feature matrix CSV, transformation pipeline, and documentation',
    context=[validate_contract_task]
)

# Task 3: Model Training
model_training_task = Task(
    description=f'''
    Train at least 2 different model types:
    
    Model 1: Random Forest
    - n_estimators: 100, 200
    - max_depth: 10, 20, None
    - min_samples_split: 2, 5
    
    Model 2: Gradient Boosting (XGBoost or LightGBM)
    - learning_rate: 0.01, 0.1
    - n_estimators: 100, 200
    - max_depth: 3, 5, 7
    
    For each model:
    1. Split data (70% train, 15% validation, 15% test)
    2. Perform 5-fold cross-validation
    3. Tune hyperparameters using GridSearchCV
    4. Train final model on train+validation
    5. Save model to {OUTPUT_DIR}/model_<name>.pkl
    
    Use random_state=42 for reproducibility
    ''',
    agent=model_trainer,
    expected_output='At least 2 trained model files (.pkl)',
    context=[feature_engineering_task]
)

# Task 4: Model Evaluation
model_evaluation_task = Task(
    description=f'''
    Evaluate and compare all trained models:
    
    For each model, calculate:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC (for classification)
    - R² and RMSE (for regression)
    - Confusion matrix
    - Feature importance
    
    Create comparison:
    1. Side-by-side metrics table
    2. ROC curves comparison
    3. Feature importance comparison
    4. Recommendation for production model
    
    Save evaluation report to {OUTPUT_DIR}/evaluation_report.md
    Save metrics to {OUTPUT_DIR}/metrics_comparison.csv
    ''',
    agent=model_evaluator,
    expected_output='Evaluation report with comprehensive model comparison',
    context=[model_training_task]
)

# Task 5: Create Model Card
create_model_card_task = Task(
    description=f'''
    Create a comprehensive Model Card following ML best practices:
    
    Sections to include:
    1. Model Details
       - Model type, version, date
       - Authors and contact
    
    2. Intended Use
       - Primary use cases
       - Out-of-scope uses
    
    3. Training Data
       - Data sources
       - Data size and split
       - Preprocessing steps
    
    4. Evaluation
       - Metrics on test set
       - Performance across subgroups
    
    5. Model Architecture
       - Algorithm details
       - Hyperparameters
    
    6. Limitations
       - Known failure modes
       - Performance boundaries
    
    7. Ethical Considerations
       - Potential biases
       - Fairness assessment
       - Environmental impact
    
    8. Recommendations
       - Deployment guidelines
       - Monitoring suggestions
    
    Save to {OUTPUT_DIR}/model_card.md
    ''',
    agent=doc_specialist,
    expected_output='Comprehensive Model Card in Markdown format',
    context=[model_evaluation_task]
)
```

### crew.py
```python
from crewai import Crew, Process
from .agents import (contract_validator, feature_engineer, 
                     model_trainer, model_evaluator, doc_specialist)
from .tasks import (validate_contract_task, feature_engineering_task,
                    model_training_task, model_evaluation_task, 
                    create_model_card_task)

class ScientistCrew:
    def __init__(self, clean_data_path, contract_path):
        self.clean_data_path = clean_data_path
        self.contract_path = contract_path
        
        self.crew = Crew(
            agents=[
                contract_validator,
                feature_engineer,
                model_trainer,
                model_evaluator,
                doc_specialist
            ],
            tasks=[
                validate_contract_task,
                feature_engineering_task,
                model_training_task,
                model_evaluation_task,
                create_model_card_task
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def kickoff(self):
        """Execute the data scientist crew"""
        print("🚀 Starting Data Scientist Crew...")
        result = self.crew.kickoff()
        print("✅ Data Scientist Crew completed!")
        return result

if __name__ == "__main__":
    crew = ScientistCrew(
        clean_data_path="artifacts/analyst/clean_data.csv",
        contract_path="artifacts/analyst/dataset_contract.json"
    )
    crew.kickoff()
```

---

## 🔄 Main Flow - Template מלא

```python
# main_flow.py
from crewai.flow.flow import Flow, listen, start
from crews.analyst_crew.crew import AnalystCrew
from crews.scientist_crew.crew import ScientistCrew
import json
import pandas as pd
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flow_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIProductFlow(Flow):
    """Main flow coordinating Data Analyst and Data Scientist crews"""
    
    @start()
    def initialize_flow(self):
        """Initialize the AI product flow"""
        logger.info("🚀 Initializing AI Product Flow")
        
        # Create artifacts directories
        Path("artifacts/analyst").mkdir(parents=True, exist_ok=True)
        Path("artifacts/scientist").mkdir(parents=True, exist_ok=True)
        
        return {
            "status": "initialized",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    @listen(initialize_flow)
    def run_analyst_crew(self, context):
        """Execute Data Analyst Crew"""
        logger.info("📊 Running Data Analyst Crew")
        
        try:
            analyst_crew = AnalystCrew()
            result = analyst_crew.kickoff()
            
            artifacts = {
                "clean_data": "artifacts/analyst/clean_data.csv",
                "contract": "artifacts/analyst/dataset_contract.json",
                "eda_report": "artifacts/analyst/eda_report.html",
                "insights": "artifacts/analyst/insights.md"
            }
            
            logger.info("✅ Data Analyst Crew completed successfully")
            return {
                **context,
                "analyst_complete": True,
                "analyst_artifacts": artifacts,
                "analyst_result": str(result)
            }
            
        except Exception as e:
            logger.error(f"❌ Data Analyst Crew failed: {str(e)}")
            raise
    
    @listen(run_analyst_crew)
    def validate_analyst_outputs(self, context):
        """Validate Data Analyst Crew outputs"""
        logger.info("✅ Validating Analyst outputs")
        
        artifacts = context["analyst_artifacts"]
        validation_results = {}
        
        # Check all files exist
        for key, path in artifacts.items():
            exists = os.path.exists(path)
            validation_results[f"{key}_exists"] = exists
            if not exists:
                raise FileNotFoundError(f"Missing required file: {path}")
            logger.info(f"✓ Found {key}: {path}")
        
        # Validate clean_data.csv
        try:
            df = pd.read_csv(artifacts["clean_data"])
            validation_results["data_rows"] = len(df)
            validation_results["data_columns"] = len(df.columns)
            validation_results["data_has_nulls"] = df.isnull().any().any()
            logger.info(f"✓ Clean data loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Failed to load clean_data.csv: {str(e)}")
        
        # Validate dataset_contract.json
        try:
            with open(artifacts["contract"]) as f:
                contract = json.load(f)
            
            # Check contract structure
            required_keys = ["schema_version", "dataset_name", "columns"]
            for key in required_keys:
                if key not in contract:
                    raise ValueError(f"Contract missing required key: {key}")
            
            # Verify columns match
            contract_columns = set(contract["columns"].keys())
            data_columns = set(df.columns)
            
            if contract_columns != data_columns:
                missing_in_contract = data_columns - contract_columns
                missing_in_data = contract_columns - data_columns
                
                error_msg = []
                if missing_in_contract:
                    error_msg.append(f"Columns in data but not in contract: {missing_in_contract}")
                if missing_in_data:
                    error_msg.append(f"Columns in contract but not in data: {missing_in_data}")
                
                raise ValueError(" | ".join(error_msg))
            
            # Validate row count
            if "row_count" in contract:
                min_rows = contract["row_count"].get("min", 0)
                if len(df) < min_rows:
                    raise ValueError(f"Data has {len(df)} rows but contract requires minimum {min_rows}")
            
            validation_results["contract_valid"] = True
            logger.info("✓ Dataset contract validated successfully")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in contract file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Contract validation failed: {str(e)}")
        
        logger.info("✅ All Analyst validations passed")
        return {
            **context,
            "analyst_validation": validation_results
        }
    
    @listen(validate_analyst_outputs)
    def run_scientist_crew(self, context):
        """Execute Data Scientist Crew"""
        logger.info("🔬 Running Data Scientist Crew")
        
        try:
            scientist_crew = ScientistCrew(
                clean_data_path=context["analyst_artifacts"]["clean_data"],
                contract_path=context["analyst_artifacts"]["contract"]
            )
            result = scientist_crew.kickoff()
            
            model_artifacts = {
                "features": "artifacts/scientist/features.csv",
                "model": "artifacts/scientist/model.pkl",
                "evaluation_report": "artifacts/scientist/evaluation_report.md",
                "model_card": "artifacts/scientist/model_card.md"
            }
            
            logger.info("✅ Data Scientist Crew completed successfully")
            return {
                **context,
                "scientist_complete": True,
                "scientist_artifacts": model_artifacts,
                "scientist_result": str(result)
            }
            
        except Exception as e:
            logger.error(f"❌ Data Scientist Crew failed: {str(e)}")
            raise
    
    @listen(run_scientist_crew)
    def validate_model_outputs(self, context):
        """Validate Data Scientist Crew outputs"""
        logger.info("✅ Validating Model outputs")
        
        artifacts = context["scientist_artifacts"]
        validation_results = {}
        
        # Check all files exist
        for key, path in artifacts.items():
            exists = os.path.exists(path)
            validation_results[f"{key}_exists"] = exists
            if not exists:
                raise FileNotFoundError(f"Missing required file: {path}")
            logger.info(f"✓ Found {key}: {path}")
        
        # Validate model can be loaded
        try:
            import joblib
            model = joblib.load(artifacts["model"])
            validation_results["model_loadable"] = True
            validation_results["model_type"] = type(model).__name__
            logger.info(f"✓ Model loaded successfully: {type(model).__name__}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # Validate model_card.md completeness
        try:
            with open(artifacts["model_card"]) as f:
                card_content = f.read()
            
            required_sections = [
                "Model Details",
                "Intended Use",
                "Training Data",
                "Performance Metrics",
                "Limitations",
                "Ethical Considerations"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in card_content:
                    missing_sections.append(section)
            
            if missing_sections:
                raise ValueError(f"Model card missing sections: {', '.join(missing_sections)}")
            
            validation_results["model_card_complete"] = True
            logger.info("✓ Model card validated successfully")
            
        except Exception as e:
            raise ValueError(f"Model card validation failed: {str(e)}")
        
        logger.info("✅ All Model validations passed")
        return {
            **context,
            "scientist_validation": validation_results,
            "final_status": "success"
        }

def run_flow():
    """Execute the complete AI product flow"""
    print("=" * 60)
    print("AI PRODUCT WORKFLOW - FINAL PROJECT")
    print("=" * 60)
    
    try:
        flow = AIProductFlow()
        result = flow.kickoff()
        
        print("\n" + "=" * 60)
        print("🎉 FLOW COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📦 Generated Artifacts:")
        print("\nData Analyst Crew:")
        for key, path in result.get("analyst_artifacts", {}).items():
            print(f"  ✓ {key}: {path}")
        
        print("\nData Scientist Crew:")
        for key, path in result.get("scientist_artifacts", {}).items():
            print(f"  ✓ {key}: {path}")
        
        print("\n✅ Validations:")
        print(f"  Analyst: {result.get('analyst_validation', {})}")
        print(f"  Scientist: {result.get('scientist_validation', {})}")
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ FLOW FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        logger.error(f"Flow execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    result = run_flow()
```

---

## 🎨 Streamlit App - Template מלא

```python
# app_streamlit.py
import streamlit as st
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Product Dashboard",
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
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "📊 EDA Report",
    "📈 Model Performance",
    "🔮 Make Predictions",
    "📋 Model Card"
])

# Load data
@st.cache_data
def load_data():
    """Load all artifacts"""
    try:
        clean_data = pd.read_csv("artifacts/analyst/clean_data.csv")
        with open("artifacts/analyst/dataset_contract.json") as f:
            contract = json.load(f)
        return clean_data, contract
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = joblib.load("artifacts/scientist/model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Dashboard Page
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🤖 AI Product Dashboard</div>', 
                unsafe_allow_html=True)
    
    data, contract = load_data()
    
    if data is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", f"{len(data):,}")
        with col2:
            st.metric("🔢 Features", len(data.columns))
        with col3:
            st.metric("🎯 Model Type", "Random Forest")
        with col4:
            st.metric("✨ Accuracy", "87%")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)

# EDA Report Page
elif page == "📊 EDA Report":
    st.title("📊 Exploratory Data Analysis")
    
    # Load and display EDA report
    eda_path = Path("artifacts/analyst/eda_report.html")
    if eda_path.exists():
        with open(eda_path) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1000, scrolling=True)
    else:
        st.warning("EDA report not found. Please run the analyst crew first.")
    
    # Also show insights
    insights_path = Path("artifacts/analyst/insights.md")
    if insights_path.exists():
        st.subheader("💡 Key Insights")
        with open(insights_path) as f:
            insights = f.read()
        st.markdown(insights)

# Model Performance Page
elif page == "📈 Model Performance":
    st.title("🎯 Model Performance Analysis")
    
    # Load evaluation report
    eval_path = Path("artifacts/scientist/evaluation_report.md")
    if eval_path.exists():
        with open(eval_path) as f:
            evaluation = f.read()
        st.markdown(evaluation)
    else:
        st.warning("Evaluation report not found.")
    
    # Metrics comparison (if available)
    metrics_path = Path("artifacts/scientist/metrics_comparison.csv")
    if metrics_path.exists():
        st.subheader("📊 Metrics Comparison")
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
        
        # Visualize metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics.plot(kind='bar', ax=ax)
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Predictions Page
elif page == "🔮 Make Predictions":
    st.title("🔮 Make Predictions")
    
    model = load_model()
    data, contract = load_data()
    
    if model is not None and data is not None:
        st.subheader("📝 Enter Feature Values")
        
        # Get feature names from contract
        feature_names = list(contract["columns"].keys())
        
        # Create input form
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            column_info = contract["columns"][feature]
            
            if i % 2 == 0:
                with col1:
                    if column_info["type"] in ["integer", "float"]:
                        value = st.number_input(
                            f"{feature}",
                            value=float(data[feature].mean()),
                            key=feature
                        )
                    else:
                        value = st.text_input(f"{feature}", key=feature)
                    input_data[feature] = value
            else:
                with col2:
                    if column_info["type"] in ["integer", "float"]:
                        value = st.number_input(
                            f"{feature}",
                            value=float(data[feature].mean()),
                            key=feature
                        )
                    else:
                        value = st.text_input(f"{feature}", key=feature)
                    input_data[feature] = value
        
        # Predict button
        if st.button("🎯 Predict", type="primary"):
            try:
                # Prepare input
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                
                # Display result
                st.success("Prediction Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", f"{max(prob)*100:.1f}%")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning("Model or data not available.")

# Model Card Page
elif page == "📋 Model Card":
    st.title("📋 Model Card")
    
    card_path = Path("artifacts/scientist/model_card.md")
    if card_path.exists():
        with open(card_path) as f:
            model_card = f.read()
        st.markdown(model_card)
    else:
        st.warning("Model card not found.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Team")
st.sidebar.markdown("Final Project - AI Development")
st.sidebar.markdown("### 📅 Date")
st.sidebar.markdown("December 2024")
```

Usethis template to get started quickly!
