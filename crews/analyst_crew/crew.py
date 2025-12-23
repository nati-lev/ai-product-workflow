# -*- coding: utf-8 -*-
"""
Data Analyst Crew
Complete crew with all 4 agents
"""

from crewai import Crew, Process
from pathlib import Path
import sys

# Add src to path for tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import our tools
from data_analysis_tools import analyze_dataset, save_validation_report
from data_cleaning_tools import clean_dataset
from eda_tools import create_eda_report
from schema_tools import create_dataset_contract

class DataAnalystCrew:
    """Data Analyst Crew - orchestrates data analysis workflow"""
    
    def __init__(self, raw_data_path='data/raw/dataset.csv', 
                 output_dir='artifacts/analyst'):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def run_validation(self):
        """Step 1: Data Validation"""
        print("\n" + "=" * 70)
        print("STEP 1: DATA VALIDATION")
        print("=" * 70)
        
        report = analyze_dataset(self.raw_data_path)
        
        if 'error' not in report:
            output_path = Path(self.output_dir) / 'validation_report.json'
            save_validation_report(report, str(output_path))
            print("\nValidation complete!")
            return report
        else:
            raise Exception("Validation failed: {}".format(report['error']))
    
    def run_cleaning(self):
        """Step 2: Data Cleaning"""
        print("\n" + "=" * 70)
        print("STEP 2: DATA CLEANING")
        print("=" * 70)
        
        clean_data_path = Path(self.output_dir) / 'clean_data.csv'
        validation_path = Path(self.output_dir) / 'validation_report.json'
        
        clean_df = clean_dataset(
            input_path=self.raw_data_path,
            output_path=str(clean_data_path),
            validation_report_path=str(validation_path)
        )
        
        print("\nCleaning complete!")
        return clean_df
    
    def run_eda(self):
        """Step 3: Exploratory Data Analysis"""
        print("\n" + "=" * 70)
        print("STEP 3: EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        clean_data_path = Path(self.output_dir) / 'clean_data.csv'
        
        report_info = create_eda_report(
            input_path=str(clean_data_path),
            output_dir=self.output_dir
        )
        
        print("\nEDA complete!")
        return report_info
    
    def run_schema_design(self):
        """Step 4: Schema Design"""
        print("\n" + "=" * 70)
        print("STEP 4: SCHEMA DESIGN")
        print("=" * 70)
        
        clean_data_path = Path(self.output_dir) / 'clean_data.csv'
        contract_path = Path(self.output_dir) / 'dataset_contract.json'
        
        contract = create_dataset_contract(
            input_path=str(clean_data_path),
            output_path=str(contract_path),
            dataset_name='Telco Customer Churn'
        )
        
        print("\nSchema design complete!")
        return contract
    
    def run(self):
        """Run the complete Data Analyst workflow"""
        print("\n" + "=" * 70)
        print("STARTING DATA ANALYST CREW")
        print("=" * 70)
        print("\nInput: {}".format(self.raw_data_path))
        print("Output: {}".format(self.output_dir))
        
        results = {}
        
        try:
            # Step 1: Validation
            results['validation'] = self.run_validation()
            
            # Step 2: Cleaning
            results['cleaning'] = self.run_cleaning()
            
            # Step 3: EDA
            results['eda'] = self.run_eda()
            
            # Step 4: Schema
            results['schema'] = self.run_schema_design()
            
            # Success summary
            print("\n" + "=" * 70)
            print("DATA ANALYST CREW - COMPLETE!")
            print("=" * 70)
            
            print("\nGenerated Artifacts:")
            print("  1. validation_report.json")
            print("  2. clean_data.csv")
            print("  3. insights.md")
            print("  4. eda plots (PNG files)")
            print("  5. dataset_contract.json")
            
            print("\nAll artifacts saved to: {}".format(self.output_dir))
            
            return results
            
        except Exception as e:
            print("\nERROR in Data Analyst Crew: {}".format(str(e)))
            raise

if __name__ == "__main__":
    print("TESTING DATA ANALYST CREW")
    print("=" * 70)
    
    # Create and run crew
    crew = DataAnalystCrew(
        raw_data_path='data/raw/dataset.csv',
        output_dir='artifacts/analyst'
    )
    
    results = crew.run()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)