"""
Dataset Selection Helper
This script helps you evaluate if a dataset is suitable for the project
"""

import pandas as pd
import numpy as np
from pathlib import Path

def evaluate_dataset(filepath: str):
    """
    Evaluate if a dataset meets project requirements
    
    Requirements:
    - At least 5,000 rows
    - At least 10 columns
    - Has a clear target variable (for prediction)
    - Contains some missing values (to demonstrate cleaning)
    - Mix of numerical and categorical features
    """
    
    print("=" * 60)
    print("DATASET EVALUATION")
    print("=" * 60)
    
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        print(f"\n✅ Successfully loaded: {filepath}")
        
        # Basic info
        print(f"\n📊 BASIC INFORMATION")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check requirements
        print(f"\n✅ REQUIREMENTS CHECK")
        
        # Requirement 1: Row count
        req1 = len(df) >= 5000
        print(f"   {'✅' if req1 else '❌'} Row count >= 5,000: {len(df):,} rows")
        
        # Requirement 2: Column count
        req2 = len(df.columns) >= 10
        print(f"   {'✅' if req2 else '❌'} Column count >= 10: {len(df.columns)} columns")
        
        # Requirement 3: Has missing values
        missing_count = df.isnull().sum().sum()
        req3 = missing_count > 0
        print(f"   {'✅' if req3 else '⚠️ '} Has missing values: {missing_count:,} nulls")
        
        # Requirement 4: Mix of types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        req4 = len(numeric_cols) >= 3 and len(categorical_cols) >= 2
        print(f"   {'✅' if req4 else '⚠️ '} Mix of types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        
        # Show column info
        print(f"\n📋 COLUMN DETAILS")
        print(f"\n   Numeric columns ({len(numeric_cols)}):")
        for col in numeric_cols[:5]:  # Show first 5
            print(f"      - {col}: {df[col].dtype}")
        if len(numeric_cols) > 5:
            print(f"      ... and {len(numeric_cols) - 5} more")
        
        print(f"\n   Categorical columns ({len(categorical_cols)}):")
        for col in categorical_cols[:5]:  # Show first 5
            unique = df[col].nunique()
            print(f"      - {col}: {unique} unique values")
        if len(categorical_cols) > 5:
            print(f"      ... and {len(categorical_cols) - 5} more")
        
        # Identify potential target variables
        print(f"\n🎯 POTENTIAL TARGET VARIABLES")
        potential_targets = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            # Binary classification targets
            if unique_count == 2:
                potential_targets.append((col, 'Binary Classification'))
            # Multi-class (3-10 classes)
            elif 3 <= unique_count <= 10 and df[col].dtype == 'object':
                potential_targets.append((col, 'Multi-class Classification'))
            # Regression (continuous numeric)
            elif unique_count > 20 and df[col].dtype in ['float64', 'int64']:
                potential_targets.append((col, 'Regression'))
        
        if potential_targets:
            for col, task_type in potential_targets[:5]:
                print(f"   - {col} ({task_type})")
        else:
            print("   ⚠️  No obvious target variable detected")
        
        # Missing values analysis
        print(f"\n🔍 MISSING VALUES ANALYSIS")
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        
        if len(missing_cols) > 0:
            print(f"   Columns with missing values:")
            for col, count in missing_cols.head(5).items():
                pct = (count / len(df)) * 100
                print(f"      - {col}: {count:,} ({pct:.1f}%)")
        else:
            print("   ⚠️  No missing values found (we can add some artificially)")
        
        # Overall assessment
        print(f"\n{'=' * 60}")
        print("OVERALL ASSESSMENT")
        print("=" * 60)
        
        all_req = req1 and req2 and req4
        if all_req:
            print("✅ This dataset is EXCELLENT for the project!")
        elif req1 and req2:
            print("✅ This dataset is SUITABLE for the project!")
        else:
            print("⚠️  This dataset may need adjustments")
        
        if not req3:
            print("💡 TIP: Consider artificially introducing some missing values")
        
        if potential_targets:
            print(f"\n🎯 Recommended task: {potential_targets[0][1]}")
            print(f"   Target variable: {potential_targets[0][0]}")
        
        print("\n" + "=" * 60)
        
        return all_req
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False

# Recommended datasets from Kaggle
RECOMMENDED_DATASETS = """
🌟 RECOMMENDED DATASETS (from Kaggle)

1. **Telco Customer Churn** (Best choice!)
   - Task: Binary Classification
   - Rows: 7,043
   - Features: 21
   - Target: Churn (Yes/No)
   - URL: kaggle.com/blastchar/telco-customer-churn
   
2. **Credit Card Fraud Detection**
   - Task: Binary Classification  
   - Rows: 284,807
   - Features: 31
   - Target: Fraud (0/1)
   - URL: kaggle.com/mlg-ulb/creditcardfraud

3. **House Prices**
   - Task: Regression
   - Rows: 1,460
   - Features: 81
   - Target: SalePrice
   - URL: kaggle.com/c/house-prices-advanced-regression-techniques

4. **Titanic**
   - Task: Binary Classification
   - Rows: 891
   - Features: 12
   - Target: Survived
   - URL: kaggle.com/c/titanic (⚠️  Small dataset)

5. **Employee Attrition**
   - Task: Binary Classification
   - Rows: 1,470
   - Features: 35
   - Target: Attrition
   - URL: kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

💡 TIP: I recommend #1 (Telco Customer Churn) - perfect size and complexity!
"""

if __name__ == "__main__":
    import sys
    
    print(RECOMMENDED_DATASETS)
    print("\n" + "=" * 60)
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        evaluate_dataset(filepath)
    else:
        print("\nUsage: python dataset_selector.py <path-to-csv>")
        print("Example: python dataset_selector.py data/raw/dataset.csv")
