# תוכנית ביצוע פרויקט - AI Multi-Agent Workflow

## 📋 סיכום המשימה

פרויקט המדמה תהליך עבודה תעשייתי של מוצר AI עם שני צוותי סוכנים:
- **Crew 1**: Data Analyst Crew - ניתוח תיאורי וניקוי נתונים
- **Crew 2**: Data Scientist Crew - מודלים חזויים
- **Flow**: אוטומציה של המעבר בין הצוותים עם ולידציות

---

## 🎯 שלב 0: תכנון והכנה (יום 1)

### 0.1 הקמת סביבת העבודה
**משימות:**
- [ ] יצירת repository ב-GitHub עם מבנה תיקיות ברור
- [ ] הגדרת `.gitignore` (Python, venv, artifacts)
- [ ] יצירת `README.md` ראשוני
- [ ] הקמת סביבה וירטואלית: `python -m venv venv`
- [ ] התקנת חבילות בסיס מ-`requirements.txt`

**בדיקה:**
- ✓ Repository נגיש לכל חברי הצוות
- ✓ כל חבר צוות יכול לעשות clone ולהריץ `pip install -r requirements.txt`

### 0.2 בחירת Dataset
**משימות:**
- [ ] חיפוש dataset מתאים ב-Kaggle/UCI
- [ ] קריטריונים: 
  - 5,000+ שורות
  - לפחות 10 עמודות
  - מתאים לבעיית חיזוי (regression/classification)
  - יש בו ערכים חסרים (לתרגול ניקוי)
- [ ] הורדה ושמירה ב-`data/raw/`

**המלצות לדאטהסטים:**
1. Customer Churn Prediction
2. Sales Forecasting
3. E-commerce Product Recommendations
4. Retail Sales Analysis

**בדיקה:**
- ✓ Dataset נטען בהצלחה ב-pandas
- ✓ יש בעיה עסקית ברורה לפתור

---

## 🔍 שלב 1: פיתוח Data Analyst Crew (ימים 2-4)

### 1.1 תכנון הסוכנים
**סוכן 1: Data Validator**
- תפקיד: בדיקת שלמות הנתונים, זיהוי בעיות
- כלים: pandas profiling, data validation checks
- פלט: `validation_report.json`

**סוכן 2: Data Cleaner**
- תפקיד: טיפול בערכים חסרים, outliers, normalization
- כלים: pandas, numpy
- פלט: `clean_data.csv`

**סוכן 3: EDA Analyst**
- תפקיד: ניתוח תיאורי, יצירת visualizations
- כלים: matplotlib, seaborn, plotly
- פלט: `eda_report.html`, `insights.md`

**סוכן 4: Schema Designer**
- תפקיד: יצירת dataset contract
- כלים: json schema
- פלט: `dataset_contract.json`

### 1.2 מימוש הסוכנים
**משימות:**
```python
# מבנה תיקיות
crews/
├── analyst_crew/
│   ├── __init__.py
│   ├── agents.py
│   ├── tasks.py
│   └── tools.py
```

**קוד לדוגמה:**
```python
# agents.py
from crewai import Agent

validator_agent = Agent(
    role='Data Validator',
    goal='Ensure data quality and identify issues',
    backstory='Expert in data quality assessment',
    verbose=True,
    allow_delegation=False
)
```

**בדיקה:**
- ✓ כל סוכן רץ בנפרד בהצלחה
- ✓ הפלטים נשמרים בתיקיית `artifacts/analyst/`

### 1.3 יצירת Dataset Contract
**מבנה ה-JSON:**
```json
{
  "schema_version": "1.0",
  "dataset_name": "clean_sales_data",
  "columns": {
    "customer_id": {
      "type": "integer",
      "nullable": false,
      "range": [1, 999999]
    },
    "purchase_amount": {
      "type": "float",
      "nullable": false,
      "range": [0, 10000]
    }
  },
  "row_count": {"min": 5000, "max": 1000000},
  "assumptions": [
    "All amounts in USD",
    "Data from 2023-2024"
  ],
  "constraints": [
    "No duplicate customer_id per transaction_id",
    "purchase_date must be valid date"
  ]
}
```

**בדיקה:**
- ✓ Contract מתאר את כל העמודות
- ✓ כולל טווחי ערכים חוקיים
- ✓ מתועד בצורה ברורה

---

## 🤖 שלב 2: פיתוח Data Scientist Crew (ימים 5-7)

### 2.1 תכנון הסוכנים
**סוכן 1: Contract Validator**
- תפקיד: וולידציה של clean_data מול dataset_contract
- פלט: `validation_status.json`

**סוכן 2: Feature Engineer**
- תפקיד: יצירת features חדשים
- דוגמאות:
  - אינטראקציות בין משתנים
  - encoding קטגוריאלי
  - feature scaling
- פלט: `features.csv`, `feature_engineering_log.md`

**סוכן 3: Model Trainer**
- תפקיד: אימון לפחות 2 מודלים
- מודלים לדוגמה:
  - Random Forest
  - Gradient Boosting (XGBoost/LightGBM)
  - Logistic Regression (baseline)
- פלט: `model_v1.pkl`, `model_v2.pkl`

**סוכן 4: Model Evaluator**
- תפקיד: השוואת מודלים והערכה
- מטריקות: accuracy, precision, recall, F1, ROC-AUC
- פלט: `evaluation_report.md`, `metrics_comparison.csv`

**סוכן 5: Documentation Specialist**
- תפקיד: יצירת Model Card
- פלט: `model_card.md`

### 2.2 מימוש Feature Engineering
**משימות:**
```python
# feature_engineering.py
def create_interaction_features(df):
    """Create feature interactions"""
    pass

def encode_categorical(df, columns):
    """One-hot or label encoding"""
    pass

def scale_numerical(df, columns):
    """StandardScaler or MinMaxScaler"""
    pass
```

**בדיקה:**
- ✓ Features נוצרים בהצלחה
- ✓ אין data leakage (train/test split מתבצע אחרי feature engineering)
- ✓ כל transformations מתועדים

### 2.3 אימון והערכת מודלים
**משימות:**
```python
# model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report

# Save model
joblib.dump(model, 'artifacts/models/model.pkl')
```

**בדיקה:**
- ✓ לפחות 2 מודלים אומנו
- ✓ Cross-validation בוצע
- ✓ מטריקות נשמרו בפורמט structured

### 2.4 יצירת Model Card
**מבנה Model Card:**
```markdown
# Model Card: Customer Churn Predictor

## Model Details
- **Model Type**: Random Forest Classifier
- **Version**: 1.0
- **Date**: December 2024
- **Author**: Data Science Team

## Intended Use
- **Primary Use**: Predict customer churn probability
- **Out-of-Scope**: Not for credit decisions

## Training Data
- **Source**: Customer database 2023-2024
- **Size**: 50,000 samples
- **Split**: 70% train, 15% validation, 15% test

## Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.87 |
| Precision | 0.84 |
| Recall | 0.82 |
| F1-Score | 0.83 |

## Limitations
- Performance degrades for customers < 3 months tenure
- Requires recalibration quarterly

## Ethical Considerations
- No demographic features used to avoid bias
- Regular fairness audits recommended
```

**בדיקה:**
- ✓ Model card מכיל כל הסעיפים הנדרשים
- ✓ מטריקות מדויקות ומתועדות

---

## 🔄 שלב 3: יצירת CrewAI Flow (ימים 8-10)

### 3.1 תכנון ה-Flow
**מבנה:**
```
Flow:
1. Start
2. Run Analyst Crew → outputs: clean_data, contract, eda
3. Validate Analyst Outputs
4. Run Data Scientist Crew → outputs: features, models, evaluation
5. Validate Model Outputs
6. End
```

### 3.2 מימוש ה-Flow
**קוד:**
```python
# main_flow.py
from crewai.flow.flow import Flow, listen, start
from crews.analyst_crew import AnalystCrew
from crews.scientist_crew import ScientistCrew
import json
import pandas as pd

class AIProductFlow(Flow):
    
    @start()
    def initialize_flow(self):
        """Initialize the flow"""
        print("🚀 Starting AI Product Flow")
        return {"status": "initialized"}
    
    @listen(initialize_flow)
    def run_analyst_crew(self, context):
        """Execute Data Analyst Crew"""
        print("📊 Running Data Analyst Crew")
        
        analyst_crew = AnalystCrew()
        result = analyst_crew.kickoff()
        
        return {
            "analyst_complete": True,
            "artifacts": {
                "clean_data": "artifacts/analyst/clean_data.csv",
                "contract": "artifacts/analyst/dataset_contract.json",
                "eda": "artifacts/analyst/eda_report.html"
            }
        }
    
    @listen(run_analyst_crew)
    def validate_analyst_outputs(self, context):
        """Validate Analyst Crew outputs"""
        print("✅ Validating Analyst outputs")
        
        artifacts = context["artifacts"]
        
        # Check files exist
        import os
        for key, path in artifacts.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
        
        # Validate contract matches data
        df = pd.read_csv(artifacts["clean_data"])
        with open(artifacts["contract"]) as f:
            contract = json.load(f)
        
        # Basic validation
        assert len(df) >= contract["row_count"]["min"]
        assert set(df.columns) == set(contract["columns"].keys())
        
        print("✓ All validations passed")
        return {"validation": "passed", **context}
    
    @listen(validate_analyst_outputs)
    def run_scientist_crew(self, context):
        """Execute Data Scientist Crew"""
        print("🔬 Running Data Scientist Crew")
        
        scientist_crew = ScientistCrew(
            clean_data_path=context["artifacts"]["clean_data"],
            contract_path=context["artifacts"]["contract"]
        )
        result = scientist_crew.kickoff()
        
        return {
            **context,
            "scientist_complete": True,
            "model_artifacts": {
                "features": "artifacts/scientist/features.csv",
                "model": "artifacts/scientist/model.pkl",
                "evaluation": "artifacts/scientist/evaluation_report.md",
                "model_card": "artifacts/scientist/model_card.md"
            }
        }
    
    @listen(run_scientist_crew)
    def validate_model_outputs(self, context):
        """Validate Data Scientist outputs"""
        print("✅ Validating Model outputs")
        
        # Check model file exists and loads
        import joblib
        model = joblib.load(context["model_artifacts"]["model"])
        
        # Verify model card completeness
        with open(context["model_artifacts"]["model_card"]) as f:
            card_content = f.read()
            required_sections = [
                "Model Details",
                "Intended Use",
                "Training Data",
                "Performance Metrics",
                "Limitations"
            ]
            for section in required_sections:
                assert section in card_content
        
        print("✓ All model validations passed")
        return {"final_status": "success", **context}

def run_flow():
    """Execute the complete flow"""
    flow = AIProductFlow()
    result = flow.kickoff()
    return result

if __name__ == "__main__":
    try:
        result = run_flow()
        print("🎉 Flow completed successfully!")
    except Exception as e:
        print(f"❌ Flow failed: {str(e)}")
        raise
```

**בדיקה:**
- ✓ Flow רץ מקצה לקצה
- ✓ Validation failures מטופלים בצורה graceful
- ✓ כל ה-artifacts נשמרים בתיקיות הנכונות

---

## 🌐 שלב 4: פיתוח ממשק משתמש (ימים 11-12)

### 4.1 אפליקציית Streamlit
**קובץ:** `app_streamlit.py`

**דפים:**
1. **Dashboard**: סטטיסטיקות כלליות
2. **EDA Report**: הצגת ניתוח תיאורי
3. **Model Performance**: מטריקות ו-visualizations
4. **Predict**: חיזוי על נתונים חדשים

**קוד לדוגמה:**
```python
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="AI Product Dashboard", layout="wide")

# Sidebar
page = st.sidebar.selectbox("Navigation", 
    ["Dashboard", "EDA Report", "Model Performance", "Predict"])

if page == "Dashboard":
    st.title("📊 AI Product Dashboard")
    
    # Load data
    df = pd.read_csv("artifacts/analyst/clean_data.csv")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Model Accuracy", "87%")
    
    st.dataframe(df.head())

elif page == "EDA Report":
    st.title("📈 Exploratory Data Analysis")
    
    # Embed HTML report
    with open("artifacts/analyst/eda_report.html") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)

elif page == "Model Performance":
    st.title("🎯 Model Performance")
    
    # Load evaluation
    with open("artifacts/scientist/evaluation_report.md") as f:
        evaluation = f.read()
    st.markdown(evaluation)

elif page == "Predict":
    st.title("🔮 Make Predictions")
    
    # Load model
    model = joblib.load("artifacts/scientist/model.pkl")
    
    # Input form
    st.subheader("Enter Features:")
    # Add input fields based on your features
    
    if st.button("Predict"):
        # Make prediction
        st.success("Prediction: Churn Risk = 75%")
```

**בדיקה:**
- ✓ אפליקציה נטענת בהצלחה
- ✓ כל הדפים עובדים
- ✓ ניתן לעשות חיזויים חדשים

### 4.2 אפליקציית Flask (אופציונלי)
**קובץ:** `app_flask.py`

**Endpoints:**
- `GET /`: דף בית
- `GET /eda`: דוח EDA
- `GET /model`: model card
- `POST /predict`: API לחיזויים

**בדיקה:**
- ✓ API endpoints עובדים
- ✓ JSON responses תקינים

---

## 📦 שלב 5: Deployment (יום 13)

### 5.1 הכנה ל-Deployment
**משימות:**
- [ ] בדיקת `requirements.txt` מעודכן
- [ ] יצירת `Procfile` (עבור Railway)
- [ ] יצירת `.streamlit/config.toml`
- [ ] בדיקת environment variables

### 5.2 Deploy ל-Streamlit Cloud
**שלבים:**
1. Push קוד ל-GitHub
2. Login ל-streamlit.io
3. New app → בחירת repo
4. Deploy!

**בדיקה:**
- ✓ אפליקציה נגישה באינטרנט
- ✓ כל הפיצ'רים עובדים

---

## 📊 שלב 6: Documentation & Presentation (ימים 14-15)

### 6.1 Documentation
**README.md מעודכן:**
```markdown
# AI Product Workflow - Final Project

## Overview
Multi-agent AI system using CrewAI for end-to-end data analysis and predictive modeling.

## Installation
```bash
git clone <repo>
cd <repo>
python -m venv venv
source venv/bin/activate  # או venv\Scripts\activate ב-Windows
pip install -r requirements.txt
```

## Usage
```bash
# Run the full flow
python main_flow.py

# Launch Streamlit app
streamlit run app_streamlit.py
```

## Project Structure
```
├── data/
│   ├── raw/
│   └── processed/
├── crews/
│   ├── analyst_crew/
│   └── scientist_crew/
├── artifacts/
│   ├── analyst/
│   └── scientist/
├── main_flow.py
├── app_streamlit.py
└── requirements.txt
```

## Team
- Member 1: Flow coordinator
- Member 2: Analyst Crew
- Member 3: Scientist Crew
- Member 4: Frontend
- Member 5: Documentation
```

### 6.2 מצגת עסקית (10-12 שקפים)
**מבנה:**
1. **Title**: שם הפרויקט + קרדיטים
2. **Business Problem**: מה הבעיה העסקית?
3. **Data Overview**: מה הדאטה שלנו?
4. **Solution Architecture**: diagram של ה-flow
5. **Crew 1 - Data Analysis**: תוצאות ו-insights
6. **Crew 2 - Predictive Models**: מודלים ומטריקות
7. **Demo**: צילומי מסך / live demo
8. **Technical Stack**: טכנולוגיות בשימוש
9. **Key Achievements**: מה הצלחנו להשיג
10. **Challenges & Learnings**: מה למדנו
11. **Future Work**: מה אפשר לשפר
12. **Q&A**: שאלות

### 6.3 סרטון Demo (≤5 דקות)
**תסריט:**
- 0:00-0:30: הקדמה לפרויקט
- 0:30-1:30: הרצת Flow (screen recording)
- 1:30-3:00: סיור באפליקציה
- 3:00-4:00: הצגת תוצאות עיקריות
- 4:00-4:45: סיכום וחידושים
- 4:45-5:00: קרדיטים

**בדיקה:**
- ✓ איכות סאונד טובה
- ✓ מסך ברור וקריא
- ✓ מתחת ל-5 דקות

---

## ✅ Checklist סופי

### קוד ו-Repository
- [ ] כל הקוד ב-GitHub עם היסטוריית commits ברורה
- [ ] Pull Requests מתועדים
- [ ] `.gitignore` מעודכן
- [ ] README.md מקיף
- [ ] requirements.txt מלא

### Artifacts
- [ ] `clean_data.csv` ✓
- [ ] `eda_report.html` ✓
- [ ] `insights.md` ✓
- [ ] `dataset_contract.json` ✓
- [ ] `features.csv` ✓
- [ ] `model.pkl` ✓
- [ ] `evaluation_report.md` ✓
- [ ] `model_card.md` ✓

### אפליקציה
- [ ] Streamlit/Flask app עובד מקומית
- [ ] Deploy מוצלח
- [ ] כל הפיצ'רים תקינים

### מסמכים
- [ ] מצגת עסקית (10-12 שקפים) ✓
- [ ] סרטון demo (≤5 דקות) ✓
- [ ] Documentation מלא ✓

---

## ⏱️ Timeline מומלץ (15 ימים)

| ימים | שלב | אחראי |
|------|-----|-------|
| 1 | הכנה וסביבת עבודה | כולם |
| 2-4 | Data Analyst Crew | חבר צוות 1,2 |
| 5-7 | Data Scientist Crew | חבר צוות 3,4 |
| 8-10 | CrewAI Flow | חבר צוות 1 |
| 11-12 | UI Development | חבר צוות 5 |
| 13 | Deployment | כולם |
| 14-15 | Documentation & Presentation | כולם |

---

## 🚨 נקודות קריטיות לתשומת לב

1. **Version Control**: commit בכל שלב משמעותי
2. **Testing**: בדיקה אחרי כל שינוי
3. **Documentation**: תיעוד תוך כדי, לא בסוף
4. **Validation**: ולידציות חזקות למניעת שגיאות
5. **Reproducibility**: random seeds, environment לוג
6. **Communication**: עדכונים יומיים בצוות

---

## 📞 תמיכה ופתרון בעיות

### בעיות נפוצות:
1. **CrewAI לא מתקין**: בדוק Python version (≥3.10)
2. **Flow נכשל**: הוסף try-except וlogger
3. **Deployment נכשל**: בדוק requirements.txt
4. **מודל לא נטען**: בדוק paths יחסיים

### משאבים:
- [CrewAI Docs](https://docs.crewai.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-Learn Docs](https://scikit-learn.org)
