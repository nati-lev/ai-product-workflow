# -*- coding: utf-8 -*-
"""
Model Training Tools
Tools for training and evaluating ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)

class ModelTrainer:
    """Model training and evaluation pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df, target_column, test_size=0.2):
        """Split data into train/test sets"""
        print("\nPreparing data...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y if len(y.unique()) < 10 else None
        )
        
        print("  Training set: {} rows".format(len(X_train)))
        print("  Test set: {} rows".format(len(X_test)))
        print("  Features: {}".format(len(X.columns)))
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression"""
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        print("  Model trained!")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        print("  Model trained!")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting"""
        print("\nTraining Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        print("  Model trained!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print("\nEvaluating {}...".format(model_name))
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print("  Accuracy: {:.4f}".format(accuracy))
        print("  Precision: {:.4f}".format(precision))
        print("  Recall: {:.4f}".format(recall))
        print("  F1 Score: {:.4f}".format(f1))
        
        # Store results
        self.results[model_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return self.results[model_name]
    
    def select_best_model(self):
        """Select the best model based on F1 score"""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        best_f1 = 0
        best_name = None
        
        for name, metrics in self.results.items():
            f1 = metrics['f1_score']
            print("\n{}:".format(name.upper()))
            print("  Accuracy: {:.4f}".format(metrics['accuracy']))
            print("  F1 Score: {:.4f}".format(metrics['f1_score']))
            
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print("\n" + "=" * 60)
        print("BEST MODEL: {}".format(best_name.upper()))
        print("F1 Score: {:.4f}".format(best_f1))
        print("=" * 60)
        
        return self.best_model, best_name
    
    def save_model(self, output_path):
        """Save the best model"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, output_path)
        print("\nModel saved to: {}".format(output_path))
    
    def train_and_evaluate(self, input_path, target_column, output_dir):
        """
        Complete training pipeline
        
        Args:
            input_path: Path to features CSV
            target_column: Name of target column
            output_dir: Directory for outputs
        """
        print("=" * 60)
        print("MODEL TRAINING & EVALUATION")
        print("=" * 60)
        
        # Load data
        print("\nLoading features: {}".format(input_path))
        df = pd.read_csv(input_path)
        print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
        
        # Check if target exists
        if target_column not in df.columns:
            # Try common variations
            possible_targets = [c for c in df.columns if 'churn' in c.lower()]
            if possible_targets:
                target_column = possible_targets[0]
                print("\nUsing target column: {}".format(target_column))
            else:
                raise ValueError("Target column '{}' not found!".format(target_column))
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)
        
        # Train models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate all models
        for name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        # Select best model
        best_model, best_name = self.select_best_model()
        
        # Save best model
        model_path = Path(output_dir) / 'model.pkl'
        self.save_model(str(model_path))
        
        # Save evaluation report
        report_path = Path(output_dir) / 'evaluation_report.json'
        report = {
            'best_model': best_name,
            'all_results': self.results,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(X_train.columns),
            'target_column': target_column
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nEvaluation report saved to: {}".format(report_path))
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        
        return best_model, report

if __name__ == "__main__":
    print("TESTING MODEL TRAINING TOOLS\n")
    
    trainer = ModelTrainer(random_state=42)
    
    # Try to find target column automatically
    df = pd.read_csv('artifacts/scientist/features.csv')
    
    # Look for target column
    target = None
    for col in df.columns:
        if 'churn' in col.lower():
            target = col
            break
    
    if not target:
        print("ERROR: Could not find target column!")
        print("Available columns:", df.columns.tolist()[:10])
        print("\nPlease specify target column manually")
    else:
        print("Found target column: {}\n".format(target))
        
        best_model, report = trainer.train_and_evaluate(
            input_path='artifacts/scientist/features.csv',
            target_column=target,
            output_dir='artifacts/scientist'
        )
        
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print("\nBest Model: {}".format(report['best_model']))
        print("Test Accuracy: {:.2%}".format(
            report['all_results'][report['best_model']]['accuracy']))