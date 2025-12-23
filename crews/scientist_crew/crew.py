# -*- coding: utf-8 -*-
"""
Data Scientist Crew
Complete ML pipeline crew
"""

from pathlib import Path
import sys
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from feature_engineering_tools import FeatureEngineer
from model_training_tools import ModelTrainer
from model_card_tools import create_model_card

class DataScientistCrew:
    """Data Scientist Crew - ML pipeline"""
    
    def __init__(self, clean_data_path='artifacts/analyst/clean_data.csv',
                 contract_path='artifacts/analyst/dataset_contract.json',
                 output_dir='artifacts/scientist',
                 target_column='churn'):
        self.clean_data_path = clean_data_path
        self.contract_path = contract_path
        self.output_dir = output_dir
        self.target_column = target_column
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def validate_contract(self):
        """Step 1: Validate data against contract"""
        print("\n" + "=" * 70)
        print("STEP 1: CONTRACT VALIDATION")
        print("=" * 70)
        
        print("\nLoading contract: {}".format(self.contract_path))
        with open(self.contract_path, 'r') as f:
            contract = json.load(f)
        
        print("Loading data: {}".format(self.clean_data_path))
        df = pd.read_csv(self.clean_data_path)
        
        print("\nContract validation:")
        print("  Expected columns: {}".format(contract['dimensions']['columns']['count']))
        print("  Actual columns: {}".format(len(df.columns)))
        print("  Expected rows: ~{}".format(contract['dimensions']['rows']['current']))
        print("  Actual rows: {}".format(len(df)))
        
        if len(df.columns) == contract['dimensions']['columns']['count']:
            print("\n  Status: PASS")
        else:
            print("\n  Status: WARNING - Column count mismatch")
        
        return True
    
    def engineer_features(self):
        """Step 2: Feature Engineering"""
        print("\n" + "=" * 70)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 70)
        
        engineer = FeatureEngineer()
        features_path = Path(self.output_dir) / 'features.csv'
        
        df_features, info = engineer.engineer_features(
            input_path=self.clean_data_path,
            output_path=str(features_path),
            target_column=self.target_column
        )
        
        return df_features, info
    
    def train_models(self):
        """Step 3: Model Training"""
        print("\n" + "=" * 70)
        print("STEP 3: MODEL TRAINING")
        print("=" * 70)
        
        trainer = ModelTrainer(random_state=42)
        features_path = Path(self.output_dir) / 'features.csv'
        
        best_model, report = trainer.train_and_evaluate(
            input_path=str(features_path),
            target_column=self.target_column,
            output_dir=self.output_dir
        )
        
        return best_model, report
    
    def create_documentation(self):
        """Step 4: Create Model Card"""
        print("\n" + "=" * 70)
        print("STEP 4: MODEL DOCUMENTATION")
        print("=" * 70)
        
        eval_report = Path(self.output_dir) / 'evaluation_report.json'
        features = Path(self.output_dir) / 'features.csv'
        model_card = Path(self.output_dir) / 'model_card.md'
        
        card = create_model_card(
            evaluation_report_path=str(eval_report),
            features_path=str(features),
            output_path=str(model_card),
            dataset_name='Telco Customer Churn'
        )
        
        return card
    
    def run(self):
        """Run complete Data Scientist workflow"""
        print("\n" + "=" * 70)
        print("STARTING DATA SCIENTIST CREW")
        print("=" * 70)
        print("\nInput: {}".format(self.clean_data_path))
        print("Output: {}".format(self.output_dir))
        print("Target: {}".format(self.target_column))
        
        results = {}
        
        try:
            # Step 1: Validate
            results['validation'] = self.validate_contract()
            
            # Step 2: Features
            df_features, feature_info = self.engineer_features()
            results['features'] = feature_info
            
            # Step 3: Training
            best_model, eval_report = self.train_models()
            results['model'] = eval_report
            
            # Step 4: Documentation
            model_card = self.create_documentation()
            results['documentation'] = 'model_card.md'
            
            # Success summary
            print("\n" + "=" * 70)
            print("DATA SCIENTIST CREW - COMPLETE!")
            print("=" * 70)
            
            print("\nGenerated Artifacts:")
            print("  1. features.csv - Engineered features")
            print("  2. model.pkl - Trained model")
            print("  3. evaluation_report.json - Model metrics")
            print("  4. model_card.md - Complete documentation")
            
            print("\nBest Model: {}".format(eval_report['best_model']))
            print("Test Accuracy: {:.2%}".format(
                eval_report['all_results'][eval_report['best_model']]['accuracy']))
            
            print("\nAll artifacts saved to: {}".format(self.output_dir))
            
            return results
            
        except Exception as e:
            print("\nERROR in Data Scientist Crew: {}".format(str(e)))
            raise

if __name__ == "__main__":
    print("TESTING DATA SCIENTIST CREW")
    print("=" * 70)
    
    # Auto-detect target column
    df = pd.read_csv('artifacts/analyst/clean_data.csv')
    target = None
    for col in df.columns:
        if 'churn' in col.lower():
            target = col
            break
    
    if not target:
        print("ERROR: Could not find target column!")
    else:
        print("Target column: {}\n".format(target))
        
        crew = DataScientistCrew(
            clean_data_path='artifacts/analyst/clean_data.csv',
            contract_path='artifacts/analyst/dataset_contract.json',
            output_dir='artifacts/scientist',
            target_column=target
        )
        
        results = crew.run()
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE!")
        print("=" * 70)