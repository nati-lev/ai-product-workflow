# -*- coding: utf-8 -*-
"""
Tools for Data Analyst Crew
Simple data analysis tools using pandas
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any

def analyze_dataset(filepath):
    """
    Analyze a dataset and return validation information
    """
    try:
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Basic info
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Column analysis
        columns_info = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_pct = float((missing_count / total_rows) * 100)
            
            columns_info[col] = {
                "type": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_percentage": round(missing_pct, 2)
            }
        
        # Duplicates
        duplicate_count = int(df.duplicated().sum())
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Create summary
        total_missing = df.isnull().sum().sum()
        summary = "Dataset has {} rows and {} columns. ".format(total_rows, total_columns)
        
        if total_missing > 0:
            missing_cols = len([c for c in columns_info.values() if c['missing_count'] > 0])
            summary += "Found {} missing values across {} columns. ".format(total_missing, missing_cols)
        else:
            summary += "No missing values found. "
            
        if duplicate_count > 0:
            summary += "Found {} duplicate rows. ".format(duplicate_count)
        else:
            summary += "No duplicate rows. "
        
        summary += "Dataset has {} numeric columns.".format(len(numeric_cols))
        
        # Create report
        report = {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": columns_info,
            "duplicates": duplicate_count,
            "numeric_columns": numeric_cols,
            "summary": summary
        }
        
        return report
        
    except Exception as e:
        return {
            "error": str(e),
            "summary": "Failed to analyze dataset: {}".format(str(e))
        }

def save_validation_report(report, output_path):
    """
    Save validation report to JSON file
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Validation report saved to: {}".format(output_path))
        return True
        
    except Exception as e:
        print("Error saving report: {}".format(str(e)))
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA ANALYSIS TOOLS")
    print("=" * 60)
    
    # Test analyze_dataset
    print("\nAnalyzing dataset...")
    report = analyze_dataset('data/raw/dataset.csv')
    
    if 'error' not in report:
        print("\nAnalysis complete!")
        print("   Rows: {:,}".format(report['total_rows']))
        print("   Columns: {}".format(report['total_columns']))
        print("   Duplicates: {}".format(report['duplicates']))
        print("\nSummary:")
        print("   {}".format(report['summary']))
        
        # Test save function
        print("\nSaving report...")
        success = save_validation_report(report, 'artifacts/analyst/validation_report.json')
        
        if success:
            print("\nTest successful! Tool is working!")
    else:
        print("\nError: {}".format(report['error']))
        print("\nMake sure data/raw/dataset.csv exists!")
