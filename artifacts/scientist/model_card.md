# Model Card - Telco Customer Churn

## Model Details

**Model Type:** Gradient Boosting
**Task:** Binary Classification (Customer Churn Prediction)
**Framework:** scikit-learn
**Created:** 2025-12-23
**Version:** 1.0

### Model Description

This model predicts customer churn using gradient boosting algorithm. The model was trained on 5634 samples 
and evaluated on 1409 samples, achieving an accuracy of 80.84%.

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

**Dataset:** Telco Customer Churn
**Training Samples:** 5,634
**Test Samples:** 1,409
**Features:** 23
**Target Variable:** churn

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
| Accuracy | 0.8084 (80.84%) |
| Precision | 0.7991 (79.91%) |
| Recall | 0.8084 (80.84%) |
| F1 Score | 0.8007 (80.07%) |

### Model Comparison

All models trained and compared:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 0.7885 | 0.7827 |
| Random Forest | 0.7991 | 0.7887 |
| Gradient Boosting (BEST) | 0.8084 | 0.8007 |

### Confusion Matrix (Best Model)

The confusion matrix for the best model (Gradient Boosting):
```
[[939, 96], [174, 200]]
```

---

## Limitations

### Known Limitations

1. **Data Limitations**
   - Model trained on historical data; performance may degrade over time
   - Limited to features available in training data
   - May not generalize to different customer segments

2. **Performance Limitations**
   - Accuracy of 80.84% means approximately 19.2% of predictions may be incorrect
   - Performance may vary across different customer demographics
   - False positives/negatives should be considered in business decisions

3. **Technical Limitations**
   - Requires feature preprocessing pipeline
   - Model size: ~23 features
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
   - Trigger retraining if accuracy drops below 72.75%
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
**Last Updated:** 2025-12-23
**Next Review:** 2025-12-23

### Changelog

**v1.0** (2025-12-23)
- Initial model training and deployment
- 23 features engineered
- Best model: Gradient Boosting with 80.84% accuracy

---

## Additional Information

### Feature Importance

Top features contributing to predictions:
- Review feature_engineering output for complete list
- 23 total features used

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
