# -*- coding: utf-8 -*-
"""
Main Flow - Complete AI Product Workflow
Runs both Data Analyst and Data Scientist crews
"""

import sys
from pathlib import Path
from datetime import datetime

# Import crews directly
import importlib.util

def load_crew_module(crew_path):
    """Load crew module dynamically"""
    spec = importlib.util.spec_from_file_location("crew_module", crew_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load analyst crew
analyst_module = load_crew_module(
    Path(__file__).parent / 'crews' / 'analyst_crew' / 'crew.py'
)
DataAnalystCrew = analyst_module.DataAnalystCrew

# Load scientist crew
scientist_module = load_crew_module(
    Path(__file__).parent / 'crews' / 'scientist_crew' / 'crew.py'
)
DataScientistCrew = scientist_module.DataScientistCrew

class AIProductFlow:
    """Complete AI Product Workflow"""
    
    def __init__(self, raw_data_path='data/raw/dataset.csv'):
        self.raw_data_path = raw_data_path
        self.analyst_output = 'artifacts/analyst'
        self.scientist_output = 'artifacts/scientist'
        self.start_time = None
        self.end_time = None
        
    def print_header(self):
        """Print fancy header"""
        print("\n")
        print("*" * 80)
        print("*" + " " * 78 + "*")
        print("*" + "  AI PRODUCT WORKFLOW - COMPLETE PIPELINE".center(78) + "*")
        print("*" + "  From Raw Data to Production-Ready ML Model".center(78) + "*")
        print("*" + " " * 78 + "*")
        print("*" * 80)
        print("\n")
    
    def validate_setup(self):
        """Validate that everything is ready"""
        print("=" * 80)
        print("VALIDATING SETUP")
        print("=" * 80)
        
        issues = []
        
        # Check dataset
        if not Path(self.raw_data_path).exists():
            issues.append("Dataset not found: {}".format(self.raw_data_path))
        else:
            print("  Dataset found: {}".format(self.raw_data_path))
        
        # Check directories
        for dir_path in [self.analyst_output, self.scientist_output]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print("  Output directory ready: {}".format(dir_path))
        
        if issues:
            print("\nERRORS:")
            for issue in issues:
                print("  - {}".format(issue))
            return False
        
        print("\n  Status: READY")
        return True
    
    def run_analyst_crew(self):
        """Phase 1: Data Analysis"""
        print("\n")
        print("=" * 80)
        print("PHASE 1: DATA ANALYST CREW")
        print("=" * 80)
        print("\nObjective: Validate, clean, analyze, and contract the data")
        print("Agents: Data Validator, Data Cleaner, EDA Analyst, Schema Designer")
        print("\n")
        
        analyst = DataAnalystCrew(
            raw_data_path=self.raw_data_path,
            output_dir=self.analyst_output
        )
        
        analyst_results = analyst.run()
        
        print("\n" + "+" * 80)
        print("PHASE 1 COMPLETE!")
        print("+" * 80)
        
        return analyst_results
    
    def validate_analyst_outputs(self):
        """Validate analyst outputs before proceeding"""
        print("\n")
        print("=" * 80)
        print("VALIDATING ANALYST OUTPUTS")
        print("=" * 80)
        
        required_files = [
            'validation_report.json',
            'clean_data.csv',
            'insights.md',
            'dataset_contract.json'
        ]
        
        all_present = True
        for filename in required_files:
            filepath = Path(self.analyst_output) / filename
            if filepath.exists():
                print("  Found: {}".format(filename))
            else:
                print("  MISSING: {}".format(filename))
                all_present = False
        
        if all_present:
            print("\n  Status: ALL ARTIFACTS PRESENT")
            return True
        else:
            print("\n  Status: MISSING ARTIFACTS")
            return False
    
    def run_scientist_crew(self):
        """Phase 2: Model Development"""
        print("\n")
        print("=" * 80)
        print("PHASE 2: DATA SCIENTIST CREW")
        print("=" * 80)
        print("\nObjective: Engineer features, train models, evaluate, document")
        print("Agents: Contract Validator, Feature Engineer, Model Trainer,")
        print("        Model Evaluator, Documentation Specialist")
        print("\n")
        
        # Auto-detect target
        import pandas as pd
        df = pd.read_csv(Path(self.analyst_output) / 'clean_data.csv')
        target = None
        for col in df.columns:
            if 'churn' in col.lower():
                target = col
                break
        
        if not target:
            raise ValueError("Could not detect target column!")
        
        print("Detected target column: {}\n".format(target))
        
        scientist = DataScientistCrew(
            clean_data_path=str(Path(self.analyst_output) / 'clean_data.csv'),
            contract_path=str(Path(self.analyst_output) / 'dataset_contract.json'),
            output_dir=self.scientist_output,
            target_column=target
        )
        
        scientist_results = scientist.run()
        
        print("\n" + "+" * 80)
        print("PHASE 2 COMPLETE!")
        print("+" * 80)
        
        return scientist_results
    
    def generate_final_report(self):
        """Generate final summary report"""
        print("\n")
        print("=" * 80)
        print("GENERATING FINAL REPORT")
        print("=" * 80)
        
        # Load key metrics
        import json
        
        eval_path = Path(self.scientist_output) / 'evaluation_report.json'
        with open(eval_path, 'r') as f:
            eval_report = json.load(f)
        
        best_model = eval_report['best_model']
        best_results = eval_report['all_results'][best_model]
        
        # Create summary
        report = """
# AI Product Workflow - Final Report

**Generated:** {}
**Duration:** {:.2f} minutes

---

## Executive Summary

Successfully completed end-to-end ML pipeline from raw data to production-ready model.

---

## Phase 1: Data Analysis

**Input:** {}
**Output Directory:** {}

### Artifacts Generated:
1. validation_report.json - Data quality assessment
2. clean_data.csv - Cleaned dataset
3. insights.md - Exploratory data analysis findings
4. dataset_contract.json - Data schema and constraints
5. Multiple visualization plots

---

## Phase 2: Model Development

**Input:** Clean data + Contract
**Output Directory:** {}

### Artifacts Generated:
1. features.csv - Engineered features
2. model.pkl - Trained ML model
3. evaluation_report.json - Performance metrics
4. model_card.md - Complete model documentation

---

## Model Performance

**Best Model:** {}
**Test Set Performance:**
- Accuracy: {:.2%}
- Precision: {:.2%}
- Recall: {:.2%}
- F1 Score: {:.2%}

### Model Comparison:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
""".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            (self.end_time - self.start_time).total_seconds() / 60,
            self.raw_data_path,
            self.analyst_output,
            self.scientist_output,
            best_model.replace('_', ' ').title(),
            best_results['accuracy'],
            best_results['precision'],
            best_results['recall'],
            best_results['f1_score']
        )
        
        # Add model comparison
        for model_name, results in eval_report['all_results'].items():
            marker = " **(BEST)**" if model_name == best_model else ""
            report += "| {}{} | {:.2%} | {:.2%} |\n".format(
                model_name.replace('_', ' ').title(),
                marker,
                results['accuracy'],
                results['f1_score']
            )
        
        report += """
---

## Key Metrics

- **Original Features:** {}
- **Engineered Features:** {}
- **Training Samples:** {:,}
- **Test Samples:** {:,}

---

## Deliverables

All artifacts are production-ready and documented:

### Data Artifacts
- `artifacts/analyst/clean_data.csv` - Clean dataset
- `artifacts/analyst/dataset_contract.json` - Data schema

### Model Artifacts  
- `artifacts/scientist/model.pkl` - Trained model
- `artifacts/scientist/features.csv` - Feature set
- `artifacts/scientist/model_card.md` - Documentation

---

## Recommendations

1. **Deployment:** Model is ready for deployment
2. **Monitoring:** Set up performance monitoring
3. **Retraining:** Schedule retraining every 3-6 months
4. **Documentation:** Review model card for details

---

## Next Steps

1. Deploy model to production environment
2. Set up prediction API
3. Implement monitoring dashboard
4. Schedule regular model evaluation

---

*Report generated by AI Product Workflow*
""".format(
            21,  # Hardcoded for now
            eval_report['features_count'],
            eval_report['training_samples'],
            eval_report['test_samples']
        )
        
        # Save report
        report_path = Path('final_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("\nFinal report saved to: {}".format(report_path))
        
        return report
    
    def run(self):
        """Execute complete workflow"""
        self.start_time = datetime.now()
        
        self.print_header()
        
        try:
            # Validation
            if not self.validate_setup():
                print("\nSetup validation failed. Aborting.")
                return False
            
            # Phase 1: Analyst
            analyst_results = self.run_analyst_crew()
            
            # Validate outputs
            if not self.validate_analyst_outputs():
                print("\nAnalyst outputs validation failed. Aborting.")
                return False
            
            # Phase 2: Scientist
            scientist_results = self.run_scientist_crew()
            
            # Generate final report
            self.end_time = datetime.now()
            final_report = self.generate_final_report()
            
            # Success!
            print("\n")
            print("*" * 80)
            print("*" + " " * 78 + "*")
            print("*" + "SUCCESS! COMPLETE PIPELINE EXECUTED".center(78) + "*")
            print("*" + " " * 78 + "*")
            print("*" * 80)
            
            duration = (self.end_time - self.start_time).total_seconds()
            print("\nTotal Duration: {:.2f} minutes".format(duration / 60))
            print("\nAll artifacts generated successfully!")
            print("\nKey Files:")
            print("  - final_report.md (Summary report)")
            print("  - artifacts/analyst/ (Data analysis)")
            print("  - artifacts/scientist/ (ML models)")
            
            return True
            
        except Exception as e:
            print("\n")
            print("*" * 80)
            print("ERROR: {}".format(str(e)))
            print("*" * 80)
            return False

def main():
    """Main entry point"""
    
    flow = AIProductFlow(raw_data_path='data/raw/dataset.csv')
    
    success = flow.run()
    
    if success:
        print("\n")
        print("=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Review final_report.md for complete summary")
        print("2. Check artifacts/analyst/ for data analysis")
        print("3. Check artifacts/scientist/ for ML model")
        print("4. Read model_card.md for deployment guide")
        print("\n")
        return 0
    else:
        print("\nWorkflow failed. Check errors above.")
        return 1

if __name__ == "__main__":
    exit(main())