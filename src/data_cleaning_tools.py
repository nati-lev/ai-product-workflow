# -*- coding: utf-8 -*-
"""
Data Cleaning Tools
Tools for cleaning and preprocessing datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_validation_report(report_path):
    """Load the validation report"""
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print("Error loading validation report: {}".format(str(e)))
        return None

def handle_missing_values(df, strategy='auto'):
    """
    Handle missing values in the dataset
    
    Args:
        df: DataFrame
        strategy: 'auto', 'drop', 'mean', 'median', 'mode'
    
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    print("\nHandling missing values...")
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        
        if missing_count > 0:
            print("  - Column '{}': {} missing values".format(col, missing_count))
            
            if strategy == 'auto':
                # Auto strategy: median for numeric, mode for categorical
                if df[col].dtype in ['float64', 'int64']:
                    # Numeric: use median
                    fill_value = df[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)
                    print("    Filled with median: {:.2f}".format(fill_value))
                else:
                    # Categorical: use mode
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(fill_value, inplace=True)
                    print("    Filled with mode: {}".format(fill_value))
            
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                print("    Dropped rows with missing values")
    
    total_filled = df.isnull().sum().sum() - df_clean.isnull().sum().sum()
    print("\nTotal missing values handled: {}".format(total_filled))
    
    return df_clean

def remove_duplicates(df):
    """Remove duplicate rows"""
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        print("\nRemoved {} duplicate rows ({:.2f}%)".format(
            removed, (removed/initial_rows)*100))
    else:
        print("\nNo duplicate rows found")
    
    return df_clean

def detect_and_cap_outliers(df, columns=None, method='iqr'):
    """
    Detect and cap outliers using IQR method
    
    Args:
        df: DataFrame
        columns: List of columns to check (None = all numeric)
        method: 'iqr' or 'zscore'
    
    Returns:
        DataFrame with capped outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    print("\nDetecting and capping outliers...")
    
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_low = (df[col] < lower_bound).sum()
            outliers_high = (df[col] > upper_bound).sum()
            total_outliers = outliers_low + outliers_high
            
            if total_outliers > 0:
                print("  - Column '{}': {} outliers found".format(col, total_outliers))
                print("    Range: [{:.2f}, {:.2f}]".format(lower_bound, upper_bound))
                
                # Cap outliers
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                print("    Outliers capped")
    
    return df_clean

def standardize_column_names(df):
    """Standardize column names (lowercase, underscores)"""
    df_clean = df.copy()
    
    original_cols = df.columns.tolist()
    df_clean.columns = [
        col.lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('.', '_')
        for col in df.columns
    ]
    
    changed = sum(1 for old, new in zip(original_cols, df_clean.columns) if old != new)
    
    if changed > 0:
        print("\nStandardized {} column names".format(changed))
    
    return df_clean

def clean_dataset(input_path, output_path, validation_report_path=None):
    """
    Main cleaning function
    
    Args:
        input_path: Path to raw dataset
        output_path: Path for cleaned dataset
        validation_report_path: Optional path to validation report
    
    Returns:
        DataFrame (cleaned)
    """
    print("=" * 60)
    print("DATA CLEANING PROCESS")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset: {}".format(input_path))
    df = pd.read_csv(input_path)
    print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
    
    # Load validation report if provided
    if validation_report_path:
        report = load_validation_report(validation_report_path)
        if report:
            print("\nValidation Report Summary:")
            print("  {}".format(report.get('summary', 'No summary available')))
    
    # Start cleaning
    initial_rows = len(df)
    
    # Step 1: Standardize column names
    df = standardize_column_names(df)
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df, strategy='auto')
    
    # Step 4: Handle outliers (only for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        df = detect_and_cap_outliers(df, columns=numeric_cols)
    
    # Summary
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print("Initial rows: {:,}".format(initial_rows))
    print("Final rows: {:,}".format(final_rows))
    print("Rows removed: {} ({:.2f}%)".format(
        rows_removed, (rows_removed/initial_rows)*100 if initial_rows > 0 else 0))
    print("Missing values remaining: {}".format(df.isnull().sum().sum()))
    
    # Save cleaned dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("\nCleaned dataset saved to: {}".format(output_path))
    
    return df

if __name__ == "__main__":
    print("TESTING DATA CLEANING TOOLS")
    print("=" * 60)
    
    # Run cleaning
    clean_df = clean_dataset(
        input_path='data/raw/dataset.csv',
        output_path='artifacts/analyst/clean_data.csv',
        validation_report_path='artifacts/analyst/validation_report.json'
    )
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)
    print("\nCleaned dataset shape: {} rows, {} columns".format(
        len(clean_df), len(clean_df.columns)))
    print("\nYou can now use: artifacts/analyst/clean_data.csv")
