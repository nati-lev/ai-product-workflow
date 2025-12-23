# -*- coding: utf-8 -*-
"""
Schema Design Tools
Creates data contracts and schema definitions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def infer_column_schema(df, column):
    """Infer detailed schema for a column"""
    col_data = df[column]
    
    schema = {
        'name': column,
        'type': str(col_data.dtype),
        'nullable': bool(col_data.isnull().any()),
        'missing_count': int(col_data.isnull().sum()),
        'missing_percentage': float((col_data.isnull().sum() / len(df)) * 100)
    }
    
    # Numeric columns
    if col_data.dtype in ['int64', 'float64']:
        schema['data_type'] = 'numeric'
        schema['statistics'] = {
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std())
        }
        
        # Suggest constraints
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        schema['suggested_constraints'] = {
            'min_value': float(col_data.min()),
            'max_value': float(col_data.max()),
            'outlier_lower_bound': float(q1 - 1.5 * iqr),
            'outlier_upper_bound': float(q3 + 1.5 * iqr)
        }
    
    # Categorical columns
    else:
        schema['data_type'] = 'categorical'
        unique_values = col_data.unique()
        value_counts = col_data.value_counts()
        
        schema['statistics'] = {
            'unique_count': int(col_data.nunique()),
            'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
            'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        }
        
        # If few unique values, list them
        if len(unique_values) <= 20:
            schema['allowed_values'] = [str(v) for v in unique_values if pd.notna(v)]
        else:
            schema['allowed_values'] = None
            schema['note'] = 'High cardinality - {} unique values'.format(len(unique_values))
    
    return schema

def create_dataset_contract(input_path, output_path, dataset_name='Dataset'):
    """
    Create comprehensive dataset contract
    
    Args:
        input_path: Path to cleaned dataset
        output_path: Path for output JSON contract
        dataset_name: Name of the dataset
    """
    print("=" * 60)
    print("CREATING DATASET CONTRACT")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading: {}".format(input_path))
    df = pd.read_csv(input_path)
    print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
    
    # Infer schema for each column
    print("\nInferring schema...")
    columns_schema = []
    
    for col in df.columns:
        print("  - Analyzing: {}".format(col))
        col_schema = infer_column_schema(df, col)
        columns_schema.append(col_schema)
    
    # Create contract
    contract = {
        'schema_version': '1.0',
        'dataset_name': dataset_name,
        'description': 'Automatically generated data contract',
        'created_at': datetime.now().isoformat(),
        
        'dimensions': {
            'rows': {
                'min': int(len(df) * 0.8),
                'max': int(len(df) * 1.2),
                'current': int(len(df))
            },
            'columns': {
                'count': len(df.columns),
                'names': df.columns.tolist()
            }
        },
        
        'columns': columns_schema,
        
        'data_quality': {
            'completeness': float(((df.count().sum()) / (len(df) * len(df.columns))) * 100),
            'total_missing': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum())
        },
        
        'validation_rules': {
            'required_columns': df.columns.tolist(),
            'no_duplicates': True,
            'max_missing_percentage': 5.0
        }
    }
    
    # Save contract
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(contract, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CONTRACT CREATED!")
    print("=" * 60)
    print("\nSaved to: {}".format(output_path))
    print("\nContract Summary:")
    print("  - Columns: {}".format(len(columns_schema)))
    print("  - Data Quality: {:.1f}% complete".format(contract['data_quality']['completeness']))
    print("  - Missing values: {}".format(contract['data_quality']['total_missing']))
    
    return contract

def validate_against_contract(df, contract):
    """
    Validate a dataframe against a contract
    
    Args:
        df: DataFrame to validate
        contract: Contract dictionary
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "=" * 60)
    print("VALIDATING AGAINST CONTRACT")
    print("=" * 60)
    
    violations = []
    warnings = []
    
    # Check columns
    required_cols = contract['validation_rules']['required_columns']
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        violations.append("Missing required columns: {}".format(list(missing_cols)))
    
    # Check row count
    row_min = contract['dimensions']['rows']['min']
    row_max = contract['dimensions']['rows']['max']
    current_rows = len(df)
    
    if current_rows < row_min:
        warnings.append("Row count {} below minimum {}".format(current_rows, row_min))
    elif current_rows > row_max:
        warnings.append("Row count {} above maximum {}".format(current_rows, row_max))
    
    # Check duplicates
    if contract['validation_rules']['no_duplicates']:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            violations.append("Found {} duplicate rows".format(dup_count))
    
    # Check missing percentage
    max_missing = contract['validation_rules']['max_missing_percentage']
    actual_missing = ((df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
    
    if actual_missing > max_missing:
        violations.append("Missing data {:.1f}% exceeds limit {:.1f}%".format(
            actual_missing, max_missing))
    
    # Check each column
    for col_schema in contract['columns']:
        col_name = col_schema['name']
        
        if col_name not in df.columns:
            continue
        
        col_data = df[col_name]
        
        # Check type
        if str(col_data.dtype) != col_schema['type']:
            warnings.append("Column '{}' type mismatch: expected {}, got {}".format(
                col_name, col_schema['type'], col_data.dtype))
        
        # Check constraints
        if 'suggested_constraints' in col_schema:
            constraints = col_schema['suggested_constraints']
            
            if col_data.min() < constraints['min_value']:
                violations.append("Column '{}' has values below minimum".format(col_name))
            
            if col_data.max() > constraints['max_value']:
                violations.append("Column '{}' has values above maximum".format(col_name))
        
        # Check allowed values
        if col_schema.get('allowed_values'):
            invalid = set(col_data.dropna().unique()) - set(col_schema['allowed_values'])
            if invalid:
                violations.append("Column '{}' has invalid values: {}".format(
                    col_name, list(invalid)[:5]))
    
    # Summary
    is_valid = len(violations) == 0
    
    print("\nValidation Result: {}".format("PASS" if is_valid else "FAIL"))
    print("Violations: {}".format(len(violations)))
    print("Warnings: {}".format(len(warnings)))
    
    if violations:
        print("\nViolations:")
        for v in violations:
            print("  - {}".format(v))
    
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("  - {}".format(w))
    
    return {
        'is_valid': is_valid,
        'violations': violations,
        'warnings': warnings
    }

if __name__ == "__main__":
    print("TESTING SCHEMA TOOLS\n")
    
    # Create contract
    contract = create_dataset_contract(
        input_path='artifacts/analyst/clean_data.csv',
        output_path='artifacts/analyst/dataset_contract.json',
        dataset_name='Telco Customer Churn'
    )
    
    # Validate the same dataset (should pass)
    print("\n\nTesting validation...")
    df = pd.read_csv('artifacts/analyst/clean_data.csv')
    result = validate_against_contract(df, contract)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)