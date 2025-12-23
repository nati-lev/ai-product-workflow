# -*- coding: utf-8 -*-
"""
Main script to run Data Analyst Crew
"""

import sys
from pathlib import Path

# Add crews to path
sys.path.insert(0, str(Path(__file__).parent / 'crews' / 'analyst_crew'))

from crew import DataAnalystCrew

def main():
    """Main execution"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  AI PRODUCT WORKFLOW - DATA ANALYST CREW".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    # Configuration
    raw_data = 'data/raw/dataset.csv'
    output_dir = 'artifacts/analyst'
    
    # Check if dataset exists
    if not Path(raw_data).exists():
        print("\nERROR: Dataset not found at {}".format(raw_data))
        print("Please make sure the dataset exists before running.")
        return
    
    print("\nConfiguration:")
    print("  Input dataset: {}".format(raw_data))
    print("  Output directory: {}".format(output_dir))
    
    # Create and run crew
    try:
        crew = DataAnalystCrew(
            raw_data_path=raw_data,
            output_dir=output_dir
        )
        
        results = crew.run()
        
        print("\n" + "*" * 70)
        print("SUCCESS! All artifacts generated.")
        print("*" * 70)
        print("\nNext steps:")
        print("  1. Review artifacts in: {}".format(output_dir))
        print("  2. Check validation_report.json for data quality")
        print("  3. Open insights.md for key findings")
        print("  4. Review dataset_contract.json for schema")
        
    except Exception as e:
        print("\n" + "*" * 70)
        print("ERROR: {}".format(str(e)))
        print("*" * 70)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())