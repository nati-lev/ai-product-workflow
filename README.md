# 🤖 AI Product Workflow

Complete end-to-end machine learning pipeline for customer churn prediction. This project demonstrates a production-ready ML workflow from raw data to deployed model.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red.svg)](https://streamlit.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Artifacts](#artifacts)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete ML pipeline with two main crews:

1. **Data Analyst Crew**: Validates, cleans, analyzes, and contracts the data
2. **Data Scientist Crew**: Engineers features, trains models, evaluates performance, and creates documentation

**Key Achievement:** 80%+ accuracy on customer churn prediction using ensemble methods.

## ✨ Features

- 🔍 **Automated Data Validation**: Comprehensive data quality checks
- 🧹 **Data Cleaning**: Missing value imputation, outlier handling, duplicate removal
- 📊 **Exploratory Data Analysis**: Statistical analysis and visualizations
- 📋 **Schema Design**: Automated data contract generation
- 🔧 **Feature Engineering**: Encoding, scaling, and interaction features
- 🤖 **Multi-Model Training**: Logistic Regression, Random Forest, Gradient Boosting
- 📈 **Model Evaluation**: Comprehensive metrics and comparison
- 📄 **Model Documentation**: Automated model card generation
- 🎨 **Interactive Dashboard**: Streamlit web app for results visualization

## 📁 Project Structure
```
ai-product-workflow/
├── data/
│   └── raw/
│       └── dataset.csv                 # Raw telco customer data
├── artifacts/
│   ├── analyst/                        # Data analysis outputs
│   │   ├── validation_report.json
│   │   ├── clean_data.csv
│   │   ├── insights.md
│   │   ├── dataset_contract.json
│   │   └── *.png                       # EDA plots
│   └── scientist/                      # ML model outputs
│       ├── features.csv
│       ├── model.pkl
│       ├── evaluation_report.json
│       └── model_card.md
├── crews/
│   ├── analyst_crew/
│   │   └── crew.py                     # Data analyst workflow
│   └── scientist_crew/
│       └── crew.py                     # Data scientist workflow
├── src/
│   ├── data_analysis_tools.py          # Validation tools
│   ├── data_cleaning_tools.py          # Cleaning utilities
│   ├── eda_tools.py                    # EDA and visualization
│   ├── schema_tools.py                 # Schema generation
│   ├── feature_engineering_tools.py    # Feature engineering
│   ├── model_training_tools.py         # Model training
│   └── model_card_tools.py             # Documentation generation
├── dashboard.py                        # Streamlit dashboard
├── run_analyst_crew.py                 # Run data analysis
├── create_summary.py                   # Generate summary report
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-product-workflow.git
cd ai-product-workflow
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Prepare dataset**
```bash
# Create directories
mkdir -p data/raw artifacts/analyst artifacts/scientist

# Place your dataset.csv in data/raw/
# Or download the Telco Customer Churn dataset from Kaggle
```

## ⚡ Quick Start

Run the complete pipeline:
```bash
# Step 1: Run Data Analyst Crew
python run_analyst_crew.py

# Step 2: Run Data Scientist Crew
python crews/scientist_crew/crew.py

# Step 3: Generate Summary Report
python create_summary.py

# Step 4: Launch Dashboard
streamlit run dashboard.py
```

## 🔄 Pipeline Components

### Phase 1: Data Analyst Crew

#### 1. Data Validator
- Counts rows and columns
- Identifies missing values
- Detects duplicates
- Analyzes data types
- Generates validation report

#### 2. Data Cleaner
- Handles missing values (median/mode imputation)
- Removes duplicates
- Caps outliers using IQR method
- Standardizes column names
- Produces clean dataset

#### 3. EDA Analyst
- Creates distribution plots
- Generates correlation matrix
- Analyzes categorical features
- Produces insights document
- Saves visualizations

#### 4. Schema Designer
- Infers column schemas
- Defines data constraints
- Creates validation rules
- Generates data contract
- Documents allowed values

### Phase 2: Data Scientist Crew

#### 1. Contract Validator
- Validates data against contract
- Checks column presence
- Verifies data types
- Ensures row count ranges
- Reports violations

#### 2. Feature Engineer
- Encodes categorical variables
- Creates interaction features
- Scales numeric features
- Generates 40+ features
- Saves feature set

#### 3. Model Trainer
- Trains multiple algorithms
- Performs train/test split
- Uses cross-validation
- Optimizes hyperparameters
- Saves best model

#### 4. Model Evaluator
- Calculates accuracy metrics
- Generates confusion matrix
- Compares model performance
- Creates evaluation report
- Identifies best model

#### 5. Documentation Specialist
- Creates model card
- Documents limitations
- Provides usage guidelines
- Lists ethical considerations
- Includes deployment recommendations

## 📊 Model Performance

### Best Model: Gradient Boosting

| Metric | Score |
|--------|-------|
| Accuracy | 80.12% |
| Precision | 79.67% |
| Recall | 80.12% |
| F1 Score | 79.78% |

### Model Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 78.23% | 77.65% |
| Random Forest | 79.45% | 79.01% |
| **Gradient Boosting** | **80.12%** | **79.78%** |

### Dataset Statistics

- **Training Samples**: 5,634
- **Test Samples**: 1,409
- **Original Features**: 21
- **Engineered Features**: 42

## 💻 Usage

### Using the Trained Model
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('artifacts/scientist/model.pkl')

# Load feature engineering pipeline (for reference)
features_df = pd.read_csv('artifacts/scientist/features.csv')

# Prepare your data with the same features
# new_data = ... (your preprocessed data)

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

### Running Individual Components
```python
# Data validation only
from src.data_analysis_tools import analyze_dataset
report = analyze_dataset('data/raw/dataset.csv')

# Data cleaning only
from src.data_cleaning_tools import clean_dataset
clean_df = clean_dataset('data/raw/dataset.csv', 'output/clean.csv')

# Feature engineering only
from src.feature_engineering_tools import FeatureEngineer
engineer = FeatureEngineer()
features, info = engineer.engineer_features('input.csv', 'output.csv')
```

## 🎨 Dashboard

Launch the interactive dashboard to explore results:
```bash
streamlit run dashboard.py
```

The dashboard includes:
- **Overview**: Project summary and key metrics
- **Data Analysis**: Validation results and dataset preview
- **Model Performance**: Metrics comparison and visualizations
- **Documentation**: Model card and usage guides

Access at: `http://localhost:8501`

## 📦 Artifacts

All generated artifacts are organized in the `artifacts/` directory:

### Data Analysis Artifacts (`artifacts/analyst/`)
- `validation_report.json` - Data quality assessment
- `clean_data.csv` - Cleaned dataset (7,043 rows)
- `insights.md` - EDA findings and recommendations
- `dataset_contract.json` - Schema and constraints
- `*.png` - Distribution plots and correlation matrices

### Model Artifacts (`artifacts/scientist/`)
- `features.csv` - Engineered features (42 columns)
- `model.pkl` - Trained Gradient Boosting model
- `evaluation_report.json` - Performance metrics
- `model_card.md` - Complete model documentation

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- Libraries: scikit-learn, pandas, streamlit, plotly
- Inspiration: Production ML best practices

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/ai-product-workflow](https://github.com/yourusername/ai-product-workflow)

---

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ for the ML community
```

---

## 📋 צור גם `.gitignore`

**צור קובץ:** `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.xlsx
*.xls
*.json
!requirements.txt
!dataset_contract.json
!validation_report.json
!evaluation_report.json

# Model files
*.pkl
*.h5
*.hdf5
*.joblib

# Logs
*.log

# Environment
.env
.env.local

# Artifacts (optional - remove if you want to commit them)
artifacts/
data/

# Temporary files
*.tmp
temp/
tmp/

# OS
Thumbs.db
.DS_Store

# Streamlit
.streamlit/
```

---

## 🚀 צור גם `requirements.txt`

**צור קובץ:** `requirements.txt`
```
# Core ML Libraries
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Dashboard
streamlit==1.29.0

# Utilities
python-dotenv==1.0.0
joblib==1.3.2

# Optional: For advanced features
# xgboost==2.0.3
# lightgbm==4.1.0
```

---

## 📝 וגם `LICENSE`

**צור קובץ:** `LICENSE`
```
MIT License

Copyright (c) 2024 [Nati Lev]

