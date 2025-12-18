# דרישות חיצוניות ותצורה נדרשת

## 🔧 דרישות מערכת

### Python
- **גרסה נדרשת**: Python 3.10 ומעלה (מומלץ 3.11)
- **בדיקה**: `python --version`

### Git
- **גרסה**: כל גרסה מעדכנת
- **תצורה נדרשת**:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### חומרה מומלצת
- **RAM**: 8GB מינימום (16GB מומלץ)
- **דיסק**: 5GB פנוי
- **מעבד**: 4 ליבות ומעלה

---

## 🌐 חשבונות וגישות חיצוניות

### 1. GitHub
**מטרה**: Version control, collaboration, deployment source

**צעדים:**
- [ ] יצירת חשבון ב-[github.com](https://github.com)
- [ ] יצירת repository חדש (public/private)
- [ ] הוספת collaborators (חברי הצוות)
- [ ] הגדרת SSH key או Personal Access Token

**תצורה:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to GitHub
cat ~/.ssh/id_ed25519.pub
```

**בדיקה:**
```bash
ssh -T git@github.com
# Should see: "Hi username! You've successfully authenticated"
```

---

### 2. Streamlit Cloud
**מטרה**: Deployment של אפליקציית Streamlit

**צעדים:**
- [ ] יצירת חשבון ב-[streamlit.io](https://streamlit.io)
- [ ] חיבור לחשבון GitHub
- [ ] הרשאות גישה ל-repository

**תצורה נדרשה:**
```toml
# .streamlit/config.toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

**secrets.toml** (אם צריך API keys):
```toml
# .streamlit/secrets.toml
# לא לעלות ל-GitHub!
OPENAI_API_KEY = "your-key-here"
```

---

### 3. Kaggle (לדאטהסטים)
**מטרה**: הורדת datasets

**צעדים:**
- [ ] יצירת חשבון ב-[kaggle.com](https://www.kaggle.com)
- [ ] הורדת API credentials

**תצורה:**
```bash
# Download kaggle.json from Kaggle account settings
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**שימוש:**
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d <dataset-path>
```

---

### 4. Railway (אופציה ל-deployment)
**מטרה**: Deployment של Flask apps או Streamlit

**צעדים:**
- [ ] יצירת חשבון ב-[railway.app](https://railway.app)
- [ ] התקנת Railway CLI

**תצורה:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init
```

**Procfile** (עבור Flask):
```
web: gunicorn app_flask:app
```

---

## 🔑 API Keys וסודות

### OpenAI (אם משתמשים ב-LLM agents)
```bash
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
```

### משתני סביבה נוספים
```bash
# .env
FLASK_ENV=development
FLASK_APP=app_flask.py
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///data.db
LOG_LEVEL=INFO
```

---

## 📦 מבנה תיקיות נדרש

```
project-root/
│
├── .git/                       # Git repository
├── .gitignore                  # Git ignore rules
├── .env                        # Environment variables (לא לעלות!)
├── .streamlit/                 # Streamlit config
│   ├── config.toml
│   └── secrets.toml           # לא לעלות!
│
├── data/                       # Data files
│   ├── raw/                   # Original datasets
│   │   └── dataset.csv
│   └── processed/             # Processed data
│       └── clean_data.csv
│
├── crews/                      # CrewAI agents
│   ├── __init__.py
│   ├── analyst_crew/
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   ├── tasks.py
│   │   └── tools.py
│   └── scientist_crew/
│       ├── __init__.py
│       ├── agents.py
│       ├── tasks.py
│       └── tools.py
│
├── artifacts/                  # Generated outputs
│   ├── analyst/
│   │   ├── clean_data.csv
│   │   ├── eda_report.html
│   │   ├── insights.md
│   │   └── dataset_contract.json
│   └── scientist/
│       ├── features.csv
│       ├── model.pkl
│       ├── evaluation_report.md
│       └── model_card.md
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── validation.py
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   └── test_models.py
│
├── notebooks/                  # Jupyter notebooks
│   └── exploration.ipynb
│
├── main_flow.py               # Main Flow execution
├── app_streamlit.py           # Streamlit app
├── app_flask.py               # Flask app (optional)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── Procfile                   # For Railway/Heroku
└── runtime.txt                # Python version for deployment
```

---

## 🔒 .gitignore נדרש

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv/

# Data
data/raw/*
!data/raw/.gitkeep
*.csv
*.xlsx
*.json
!dataset_contract.json

# Models
artifacts/scientist/*.pkl
artifacts/scientist/*.joblib
*.h5
*.pt

# Environment
.env
.env.local
.streamlit/secrets.toml

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Deployment
dist/
build/
*.egg-info/
```

---

## 🧪 בדיקות אינטגרציה

### בדיקה 1: סביבת Python
```bash
# Check Python version
python --version  # Should be 3.10+

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Verify key packages
python -c "import crewai; print(crewai.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

### בדיקה 2: Git Repository
```bash
# Initialize repo
git init

# Add files
git add .

# First commit
git commit -m "Initial commit"

# Connect to GitHub
git remote add origin git@github.com:username/repo.git

# Push
git push -u origin main
```

### בדיקה 3: Streamlit Local
```bash
# Run app
streamlit run app_streamlit.py

# Should open browser at http://localhost:8501
```

### בדיקה 4: CrewAI Flow
```bash
# Run flow
python main_flow.py

# Check outputs
ls artifacts/analyst/
ls artifacts/scientist/
```

---

## 📊 כלי ניטור ובדיקה

### 1. Git Statistics
```bash
# View commit history
git log --oneline --graph --all

# View contributors
git shortlog -sn

# View file changes
git diff --stat
```

### 2. Code Quality
```bash
# Format code
black .

# Lint code
flake8 --max-line-length=100

# Type checking (if using types)
mypy src/
```

### 3. Testing
```bash
# Run tests
pytest tests/ -v

# With coverage
pytest --cov=src tests/

# Generate coverage report
coverage html
```

---

## 🚀 תהליך Deployment

### Streamlit Cloud Deployment

**שלב 1: הכנה**
```bash
# Ensure requirements.txt is complete
pip freeze > requirements.txt

# Create .streamlit/config.toml
# Add secrets if needed to .streamlit/secrets.toml
```

**שלב 2: Push to GitHub**
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

**שלב 3: Deploy**
1. Go to streamlit.io
2. Click "New app"
3. Select repository
4. Select `app_streamlit.py` as main file
5. Click Deploy!

**שלב 4: Monitor**
- Check logs in Streamlit dashboard
- Test all features
- Share URL with team

---

## 🆘 פתרון בעיות נפוצות

### בעיה: ModuleNotFoundError
**פתרון:**
```bash
# Ensure venv is activated
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### בעיה: CrewAI Agent לא רץ
**פתרון:**
```bash
# Check API keys
echo $OPENAI_API_KEY

# Verify agent configuration
python -c "from crews.analyst_crew.agents import validator_agent; print(validator_agent)"
```

### בעיה: Streamlit deployment נכשל
**פתרון:**
1. בדוק Python version ב-`runtime.txt`
2. בדוק שכל הקבצים נמצאים בגיט
3. בדוק logs בממשק Streamlit

### בעיה: Model לא נטען
**פתרון:**
```python
# Use absolute paths
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "scientist" / "model.pkl"

import joblib
model = joblib.load(MODEL_PATH)
```

---

## 📚 משאבים נוספים

### תיעוד רשמי
- [CrewAI Docs](https://docs.crewai.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Flask Docs](https://flask.palletsprojects.com)
- [Scikit-Learn Docs](https://scikit-learn.org/stable/)
- [Pandas Docs](https://pandas.pydata.org/docs/)

### טוטוריאלים מומלצים
- [CrewAI Getting Started](https://github.com/joaomdmoura/crewAI)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [ML Project Template](https://github.com/drivendata/cookiecutter-data-science)

### קהילות
- [CrewAI Discord](https://discord.gg/X4JWnZnxPb)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [Kaggle Forums](https://www.kaggle.com/discussion)

---

## ✅ Checklist התקנה מלא

### הכנה ראשונית
- [ ] Python 3.10+ מותקן ועובד
- [ ] Git מותקן ומוגדר
- [ ] חשבון GitHub פעיל
- [ ] חשבון Kaggle עם API credentials
- [ ] חשבון Streamlit Cloud

### סביבת פיתוח
- [ ] Repository נוצר ב-GitHub
- [ ] Clone local עבד בהצלחה
- [ ] Virtual environment נוצר
- [ ] requirements.txt הותקן במלואו
- [ ] .gitignore מוגדר נכון
- [ ] .env נוצר עם משתנים נדרשים

### מבנה פרויקט
- [ ] כל התיקיות נוצרו
- [ ] Dataset הורד ונמצא ב-data/raw/
- [ ] קבצי __init__.py בכל תיקיית Python

### בדיקות
- [ ] `python main_flow.py` רץ בלי שגיאות
- [ ] `streamlit run app_streamlit.py` עובד מקומית
- [ ] Git commits מתבצעים בהצלחה
- [ ] Push ל-GitHub עובד

### Deployment
- [ ] Streamlit app deployed ונגיש
- [ ] כל הפיצ'רים עובדים בפרודקשן
- [ ] URL משותף עם הצוות

---

**שאלות? צור קשר עם חברי הצוות או פנה למדריך הקורס.**
