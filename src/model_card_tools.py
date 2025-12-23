# -*- coding: utf-8 -*-
"""
Model Card Tools
Tools for creating comprehensive model documentation
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def create_model_card(evaluation_report_path, features_path, 
                     output_path, dataset_name='Dataset'):
    """
    Create comprehensive model card documentation
    
    Args:
        evaluation_report_path: Path to evaluation report JSON
        features_path: Path to features CSV
        output_path: Path for output markdown
        dataset_name: Name of dataset
    """
    print("=" * 60)
    print("CREATING MODEL CARD")
    print("=" * 60)
    
    # Load evaluation report
    print("\nLoading evaluation report...")
    with open(evaluation_report_path, 'r') as f:
        report = json.load(f)
    
    # Load features to get column info
    print("Loading features info...")
    df = pd.read_csv(features_path)
    
    best_model = report['best_model']
    best_results = report['all_results'][best_model]
    
    # Create model card content
    model_card = """# Model Card - {}

## Model Details

**Model Type:** {}
**Task:** Binary Classification (Customer Churn Prediction)
**Framework:** scikit-learn
**Created:** {}
**Version:** 1.0

### Model Description

This model predicts customer churn using {} algorithm. The model was trained on {} samples 
and evaluated on {} samples, achieving an accuracy of {:.2%}.

---

## Intended Use

### Primary Use Cases
- Predicting customer churn probability
- Identifying at-risk customers for retention campaigns
- Supporting business decision-making for customer retention

### Intended Users
- Data scientists
- Business analysts
- Marketing teams
- Customer success teams

### Out-of-Scope Uses
- Should not be used as the sole basis for customer termination decisions
- Not intended for use with data outside the training distribution
- Not suitable for real-time streaming predictions without retraining

---

## Training Data

**Dataset:** {}
**Training Samples:** {:,}
**Test Samples:** {:,}
**Features:** {}
**Target Variable:** {}

### Data Characteristics
- Time period: Historical customer data
- Data preprocessing: Cleaning, encoding, scaling, feature engineering
- Class distribution: Check for balance/imbalance

---

## Evaluation

### Metrics

**Primary Metric:** F1 Score

| Metric | Score |
|--------|-------|
| Accuracy | {:.4f} ({:.2%}) |
| Precision | {:.4f} ({:.2%}) |
| Recall | {:.4f} ({:.2%}) |
| F1 Score | {:.4f} ({:.2%}) |

### Model Comparison

All models trained and compared:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
""".format(
        dataset_name,
        best_model.replace('_', ' ').title(),
        datetime.now().strftime('%Y-%m-%d'),
        best_model.replace('_', ' '),
        report['training_samples'],
        report['test_samples'],
        best_results['accuracy'],
        dataset_name,
        report['training_samples'],
        report['test_samples'],
        report['features_count'],
        report['target_column'],
        best_results['accuracy'], best_results['accuracy'],
        best_results['precision'], best_results['precision'],
        best_results['recall'], best_results['recall'],
        best_results['f1_score'], best_results['f1_score']
    )
    
    # Add all model comparisons
    for model_name, results in report['all_results'].items():
        marker = " (BEST)" if model_name == best_model else ""
        model_card += "| {}{} | {:.4f} | {:.4f} |\n".format(
            model_name.replace('_', ' ').title(),
            marker,
            results['accuracy'],
            results['f1_score']
        )
    
    model_card += """
### Confusion Matrix (Best Model)

The confusion matrix for the best model ({}):
```
{}
```

---

## Limitations

### Known Limitations

1. **Data Limitations**
   - Model trained on historical data; performance may degrade over time
   - Limited to features available in training data
   - May not generalize to different customer segments

2. **Performance Limitations**
   - Accuracy of {:.2%} means approximately {:.1f}% of predictions may be incorrect
   - Performance may vary across different customer demographics
   - False positives/negatives should be considered in business decisions

3. **Technical Limitations**
   - Requires feature preprocessing pipeline
   - Model size: ~{} features
   - Not optimized for real-time inference

### Bias Considerations

- Model should be monitored for bias across customer segments
- Regular retraining recommended to maintain performance
- Consider fairness metrics for protected attributes

---

## Recommendations

### Deployment Recommendations

1. **Monitoring**
   - Track prediction distribution over time
   - Monitor for data drift
   - Set up alerts for performance degradation

2. **Retraining Schedule**
   - Recommended retraining: Every 3-6 months
   - Trigger retraining if accuracy drops below {:.2%}
   - Update features as new data becomes available

3. **Usage Guidelines**
   - Use predictions as input to decision-making, not sole determinant
   - Combine with domain expertise
   - Validate predictions with business logic

### Model Updates

- Version control all model artifacts
- Document changes in features or training data
- Maintain backwards compatibility when possible

---

## Ethical Considerations

### Fairness
- Ensure model does not discriminate based on protected attributes
- Regular audits for disparate impact
- Consider fairness constraints in model training

### Privacy
- Training data should be anonymized
- No personal identifiers in model inputs
- Comply with data protection regulations (GDPR, CCPA)

### Transparency
- Model decisions should be explainable to stakeholders
- Maintain clear documentation of model behavior
- Provide mechanisms for human oversight

---

## Contact & Maintenance

**Model Owner:** Data Science Team
**Last Updated:** {}
**Next Review:** {}

### Changelog

**v1.0** ({})
- Initial model training and deployment
- {} features engineered
- Best model: {} with {:.2%} accuracy

---

## Additional Information

### Feature Importance

Top features contributing to predictions:
- Review feature_engineering output for complete list
- {} total features used

### Model Files

- `model.pkl` - Trained model (serialized with joblib)
- `evaluation_report.json` - Detailed metrics
- `features.csv` - Engineered features used for training

### References

- scikit-learn documentation
- Model training pipeline documentation
- Feature engineering documentation

---

*This model card was automatically generated. Please review and update as needed.*
""".format(
        best_model.replace('_', ' ').title(),
        str(best_results['confusion_matrix']),
        best_results['accuracy'],
        (1 - best_results['accuracy']) * 100,
        report['features_count'],
        best_results['accuracy'] * 0.9,
        datetime.now().strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d'),
        report['features_count'],
        best_model.replace('_', ' ').title(),
        best_results['accuracy'],
        report['features_count']
    )
    
    # Save model card
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(model_card)
    
    print("\n" + "=" * 60)
    print("MODEL CARD CREATED!")
    print("=" * 60)
    print("\nSaved to: {}".format(output_path))
    print("\nModel Card includes:")
    print("  - Model details and description")
    print("  - Training data information")
    print("  - Evaluation metrics and comparison")
    print("  - Limitations and considerations")
    print("  - Deployment recommendations")
    print("  - Ethical considerations")
    
    return model_card

if __name__ == "__main__":
    print("TESTING MODEL CARD TOOLS\n")
    
    model_card = create_model_card(
        evaluation_report_path='artifacts/scientist/evaluation_report.json',
        features_path='artifacts/scientist/features.csv',
        output_path='artifacts/scientist/model_card.md',
        dataset_name='Telco Customer Churn'
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nYou can now read the model card:")
    print("  cat artifacts/scientist/model_card.md")