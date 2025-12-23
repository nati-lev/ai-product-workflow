# -*- coding: utf-8 -*-
"""
Feature Engineering Tools
Tools for creating and transforming features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        
    def encode_categorical(self, df, columns=None):
        """Encode categorical columns"""
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        print("\nEncoding categorical features...")
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print("  - Encoded: {}".format(col))
        
        return df_encoded
    
    def create_interaction_features(self, df, numeric_cols):
        """Create interaction features between numeric columns"""
        df_feat = df.copy()
        
        print("\nCreating interaction features...")
        
        if len(numeric_cols) >= 2:
            # Create some interactions
            for i in range(len(numeric_cols)):
                for j in range(i + 1, min(i + 3, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # Multiplication
                    new_col = '{}_{}_mult'.format(col1, col2)
                    df_feat[new_col] = df_feat[col1] * df_feat[col2]
                    print("  - Created: {}".format(new_col))
        
        return df_feat
    
    def scale_features(self, df, columns):
        """Scale numeric features"""
        df_scaled = df.copy()
        
        print("\nScaling features...")
        
        self.scaler = StandardScaler()
        df_scaled[columns] = self.scaler.fit_transform(df[columns])
        
        print("  - Scaled {} features".format(len(columns)))
        
        return df_scaled
    
    def engineer_features(self, input_path, output_path, target_column=None):
        """
        Main feature engineering pipeline
        
        Args:
            input_path: Path to clean data
            output_path: Path for engineered features
            target_column: Name of target column (optional)
        """
        print("=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # Load data
        print("\nLoading: {}".format(input_path))
        df = pd.read_csv(input_path)
        print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature engineering if specified
        if target_column:
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        print("\nNumeric columns: {}".format(len(numeric_cols)))
        print("Categorical columns: {}".format(len(categorical_cols)))
        
        # Start with original data
        df_engineered = df.copy()
        
        # Encode categorical
        df_engineered = self.encode_categorical(df_engineered, categorical_cols)
        
        # Create interactions (only for first few numeric cols to avoid explosion)
        if len(numeric_cols) > 0:
            df_engineered = self.create_interaction_features(
                df_engineered, numeric_cols[:3])
        
        # Scale numeric features
        numeric_to_scale = [c for c in df_engineered.columns 
                           if df_engineered[c].dtype in ['float64', 'int64']
                           and c != target_column]
        
        if len(numeric_to_scale) > 0:
            df_engineered = self.scale_features(df_engineered, numeric_to_scale)
        
        # Save engineered features
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_engineered.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE!")
        print("=" * 60)
        print("\nOriginal features: {}".format(len(df.columns)))
        print("Engineered features: {}".format(len(df_engineered.columns)))
        print("New features created: {}".format(len(df_engineered.columns) - len(df.columns)))
        print("\nSaved to: {}".format(output_path))
        
        # Save feature info
        feature_info = {
            'original_features': len(df.columns),
            'engineered_features': len(df_engineered.columns),
            'new_features': len(df_engineered.columns) - len(df.columns),
            'feature_names': df_engineered.columns.tolist(),
            'numeric_features': numeric_to_scale,
            'categorical_encoded': list(self.label_encoders.keys())
        }
        
        return df_engineered, feature_info

if __name__ == "__main__":
    print("TESTING FEATURE ENGINEERING TOOLS\n")
    
    engineer = FeatureEngineer()
    
    df_features, info = engineer.engineer_features(
        input_path='artifacts/analyst/clean_data.csv',
        output_path='artifacts/scientist/features.csv',
        target_column='churn'  # Adjust based on your target
    )
    
    print("\nFeature Info:")
    print("  Original: {}".format(info['original_features']))
    print("  Engineered: {}".format(info['engineered_features']))
    print("  New: {}".format(info['new_features']))
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)