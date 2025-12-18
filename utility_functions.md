# Utility Functions - פונקציות עזר מוכנות לשימוש

## 📁 src/data_processing.py

```python
"""
Data processing utilities for the AI Product Workflow
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load CSV file with error handling
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise

def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV with directory creation
    
    Args:
        df: DataFrame to save
        filepath: Destination path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")

def detect_missing_values(df: pd.DataFrame) -> Dict:
    """
    Analyze missing values in DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with missing value analysis
    """
    missing_info = {}
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_info[col] = {
                "count": int(missing_count),
                "percentage": float(missing_count / len(df) * 100),
                "dtype": str(df[col].dtype)
            }
    
    return missing_info

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str] = None) -> Dict:
    """
    Detect outliers using IQR method
    
    Args:
        df: Input DataFrame
        columns: List of numeric columns (None = all numeric)
        
    Returns:
        Dictionary with outlier information
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            outliers_info[col] = {
                "count": len(outliers),
                "percentage": float(len(outliers) / len(df) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "min_value": float(df[col].min()),
                "max_value": float(df[col].max())
            }
    
    return outliers_info

def handle_missing_values(df: pd.DataFrame, strategy: Dict = None) -> pd.DataFrame:
    """
    Handle missing values based on strategy
    
    Args:
        df: Input DataFrame
        strategy: Dictionary mapping columns to strategies
                 {'col1': 'mean', 'col2': 'median', 'col3': 'mode', 'col4': 'drop'}
                 
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy is None:
        # Default strategy: median for numeric, mode for categorical
        strategy = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    strategy[col] = 'median'
                else:
                    strategy[col] = 'mode'
    
    for col, method in strategy.items():
        if col not in df_clean.columns:
            continue
            
        if method == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif method == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif method == 'mode':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif method == 'ffill':
            df_clean[col].fillna(method='ffill', inplace=True)
        elif method == 'bfill':
            df_clean[col].fillna(method='bfill', inplace=True)
        else:
            # Fill with specific value
            df_clean[col].fillna(method, inplace=True)
    
    logger.info(f"Handled missing values. Rows: {len(df)} -> {len(df_clean)}")
    return df_clean

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates (None = all columns)
        
    Returns:
        DataFrame without duplicates
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates(subset=subset)
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
    
    return df_clean

def cap_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Cap outliers at boundaries
    
    Args:
        df: Input DataFrame
        columns: Columns to process
        method: 'iqr' or 'percentile'
        
    Returns:
        DataFrame with capped outliers
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        else:  # percentile
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
        
        df_clean[col] = df_clean[col].clip(lower, upper)
        
        logger.info(f"Capped outliers in {col}: [{lower:.2f}, {upper:.2f}]")
    
    return df_clean

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names (lowercase, underscores)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df_clean = df.copy()
    df_clean.columns = [
        col.lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('.', '_')
        for col in df.columns
    ]
    return df_clean

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive data summary
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": detect_missing_values(df),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary
```

---

## 🔧 src/feature_engineering.py

```python
"""
Feature engineering utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def create_interaction_features(self, 
                                   df: pd.DataFrame, 
                                   pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features (multiplication)
        
        Args:
            df: Input DataFrame
            pairs: List of column pairs to interact
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        
        for col1, col2 in pairs:
            feature_name = f"{col1}_x_{col2}"
            df_new[feature_name] = df[col1] * df[col2]
            logger.info(f"Created interaction feature: {feature_name}")
        
        return df_new
    
    def create_polynomial_features(self,
                                  df: pd.DataFrame,
                                  columns: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        df_new = df.copy()
        
        for col in columns:
            for d in range(2, degree + 1):
                feature_name = f"{col}_pow{d}"
                df_new[feature_name] = df[col] ** d
                logger.info(f"Created polynomial feature: {feature_name}")
        
        return df_new
    
    def encode_categorical(self,
                          df: pd.DataFrame,
                          columns: List[str],
                          method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: Columns to encode
            method: 'onehot' or 'label'
            
        Returns:
            DataFrame with encoded features
        """
        df_new = df.copy()
        
        for col in columns:
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_new = pd.concat([df_new, dummies], axis=1)
                df_new = df_new.drop(columns=[col])
                logger.info(f"One-hot encoded {col}: {len(dummies.columns)} features")
                
            elif method == 'label':
                # Label encoding
                le = LabelEncoder()
                df_new[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded {col}: {len(le.classes_)} classes")
        
        return df_new
    
    def scale_features(self,
                      df: pd.DataFrame,
                      columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: 'standard' or 'minmax'
            
        Returns:
            DataFrame with scaled features
        """
        df_new = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        df_new[columns] = scaler.fit_transform(df[columns])
        
        for col in columns:
            self.scalers[col] = scaler
        
        logger.info(f"Scaled {len(columns)} features using {method}")
        return df_new
    
    def create_date_features(self,
                           df: pd.DataFrame,
                           date_column: str) -> pd.DataFrame:
        """
        Extract features from date column
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with date features
        """
        df_new = df.copy()
        df_new[date_column] = pd.to_datetime(df_new[date_column])
        
        df_new[f'{date_column}_year'] = df_new[date_column].dt.year
        df_new[f'{date_column}_month'] = df_new[date_column].dt.month
        df_new[f'{date_column}_day'] = df_new[date_column].dt.day
        df_new[f'{date_column}_dayofweek'] = df_new[date_column].dt.dayofweek
        df_new[f'{date_column}_quarter'] = df_new[date_column].dt.quarter
        df_new[f'{date_column}_is_weekend'] = df_new[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        logger.info(f"Created 6 date features from {date_column}")
        return df_new
    
    def handle_skewness(self,
                       df: pd.DataFrame,
                       columns: List[str],
                       threshold: float = 0.75) -> pd.DataFrame:
        """
        Apply log transformation to skewed features
        
        Args:
            df: Input DataFrame
            columns: Columns to check
            threshold: Skewness threshold
            
        Returns:
            DataFrame with transformed features
        """
        df_new = df.copy()
        
        for col in columns:
            skew = df[col].skew()
            if abs(skew) > threshold:
                df_new[f'{col}_log'] = np.log1p(df[col])
                logger.info(f"Log transformed {col} (skew: {skew:.2f})")
        
        return df_new
    
    def save_transformers(self, filepath: str):
        """Save all transformers"""
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(transformers, filepath)
        logger.info(f"Saved transformers to {filepath}")
    
    def load_transformers(self, filepath: str):
        """Load transformers"""
        transformers = joblib.load(filepath)
        self.scalers = transformers['scalers']
        self.encoders = transformers['encoders']
        self.feature_names = transformers['feature_names']
        logger.info(f"Loaded transformers from {filepath}")
```

---

## 🤖 src/model_training.py

```python
"""
Model training utilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import joblib
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training and evaluation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.results = {}
    
    def split_data(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   test_size: float = 0.2,
                   val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, 
            random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           params: Dict = None) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            
        Returns:
            Trained model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("Trained Random Forest model")
        return model
    
    def train_gradient_boosting(self,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               params: Dict = None) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            
        Returns:
            Trained model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': self.random_state
            }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = model
        logger.info("Trained Gradient Boosting model")
        return model
    
    def tune_hyperparameters(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            model_type: str = 'random_forest') -> Dict:
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'random_forest' or 'gradient_boosting'
            
        Returns:
            Best parameters
        """
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            model = GradientBoostingClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def evaluate_model(self,
                      model,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name for results storage
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        self.results[model_name] = metrics
        logger.info(f"Evaluated {model_name}: F1={metrics['f1_score']:.4f}")
        
        return metrics
    
    def cross_validate(self,
                      model,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        results = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        return results
    
    def get_feature_importance(self,
                              model,
                              feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            feat_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            return feat_imp
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
    
    def save_model(self, model, filepath: str):
        """Save trained model"""
        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return model
    
    def select_best_model(self) -> Tuple[str, Dict]:
        """
        Select best model based on F1 score
        
        Returns:
            Tuple of (model_name, metrics)
        """
        best_model_name = max(self.results, key=lambda k: self.results[k]['f1_score'])
        best_metrics = self.results[best_model_name]
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} (F1: {best_metrics['f1_score']:.4f})")
        
        return best_model_name, best_metrics
```

---

## ✅ src/validation.py

```python
"""
Validation utilities for dataset contracts and model outputs
"""
import pandas as pd
import json
from typing import Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ContractValidator:
    """Validate data against dataset contracts"""
    
    def __init__(self, contract_path: str):
        """
        Initialize validator with contract
        
        Args:
            contract_path: Path to contract JSON file
        """
        with open(contract_path) as f:
            self.contract = json.load(f)
        logger.info(f"Loaded contract from {contract_path}")
    
    def validate_schema(self, df: pd.DataFrame) -> Dict:
        """
        Validate DataFrame schema against contract
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'schema_valid': True,
            'errors': [],
            'warnings': []
        }
        
        contract_columns = set(self.contract['columns'].keys())
        data_columns = set(df.columns)
        
        # Check missing columns
        missing = contract_columns - data_columns
        if missing:
            results['schema_valid'] = False
            results['errors'].append(f"Missing columns: {missing}")
        
        # Check extra columns
        extra = data_columns - contract_columns
        if extra:
            results['warnings'].append(f"Extra columns not in contract: {extra}")
        
        # Validate data types
        for col in contract_columns & data_columns:
            expected_type = self.contract['columns'][col]['type']
            actual_type = str(df[col].dtype)
            
            type_map = {
                'integer': ['int64', 'int32'],
                'float': ['float64', 'float32'],
                'string': ['object'],
                'boolean': ['bool']
            }
            
            if actual_type not in type_map.get(expected_type, []):
                results['warnings'].append(
                    f"Column {col}: expected {expected_type}, got {actual_type}"
                )
        
        return results
    
    def validate_constraints(self, df: pd.DataFrame) -> Dict:
        """
        Validate data constraints
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results
        """
        results = {
            'constraints_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check row count
        if 'row_count' in self.contract:
            min_rows = self.contract['row_count'].get('min', 0)
            max_rows = self.contract['row_count'].get('max', float('inf'))
            
            if len(df) < min_rows:
                results['constraints_valid'] = False
                results['errors'].append(
                    f"Row count {len(df)} below minimum {min_rows}"
                )
            if len(df) > max_rows:
                results['warnings'].append(
                    f"Row count {len(df)} exceeds maximum {max_rows}"
                )
        
        # Check column constraints
        for col, spec in self.contract['columns'].items():
            if col not in df.columns:
                continue
            
            # Check nullable
            if not spec.get('nullable', True):
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results['constraints_valid'] = False
                    results['errors'].append(
                        f"Column {col} has {null_count} null values but is non-nullable"
                    )
            
            # Check value ranges
            if 'range' in spec and df[col].dtype in ['int64', 'float64']:
                min_val, max_val = spec['range']
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    results['warnings'].append(
                        f"Column {col} has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                    )
            
            # Check allowed values
            if 'allowed_values' in spec:
                invalid = df[~df[col].isin(spec['allowed_values'])]
                if len(invalid) > 0:
                    results['constraints_valid'] = False
                    results['errors'].append(
                        f"Column {col} has {len(invalid)} invalid values"
                    )
        
        return results
    
    def validate_all(self, df: pd.DataFrame) -> Dict:
        """
        Run all validations
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Complete validation results
        """
        schema_results = self.validate_schema(df)
        constraint_results = self.validate_constraints(df)
        
        all_valid = schema_results['schema_valid'] and constraint_results['constraints_valid']
        
        results = {
            'overall_valid': all_valid,
            'schema': schema_results,
            'constraints': constraint_results,
            'summary': {
                'total_errors': len(schema_results['errors']) + len(constraint_results['errors']),
                'total_warnings': len(schema_results['warnings']) + len(constraint_results['warnings'])
            }
        }
        
        if all_valid:
            logger.info("✅ All validations passed")
        else:
            logger.error(f"❌ Validation failed: {results['summary']['total_errors']} errors")
        
        return results

def save_validation_report(results: Dict, filepath: str):
    """
    Save validation results to JSON
    
    Args:
        results: Validation results dictionary
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved validation report to {filepath}")
```

כל הפונקציות האלה מוכנות לשימוש ומתועדות היטב!
