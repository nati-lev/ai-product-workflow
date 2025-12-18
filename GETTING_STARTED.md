# 🚀 התחלה צעד אחר צעד - מדריך מפורט

## 📋 איפה אנחנו עכשיו?

סיימנו את **שלב 0.1** - יצירת הקבצים הבסיסיים.  
יש לך עכשיו:
- ✅ מבנה תיקיות מוכן
- ✅ `.gitignore` מלא
- ✅ `README.md` ראשוני
- ✅ `requirements.txt` עם כל החבילות
- ✅ סקריפטים להורדת dataset

---

## 🎯 השלבים הבאים (עכשיו!)

### צעד 1: הקמת סביבת העבודה (5 דקות)

בחלון הטרמינל שלך, הרץ:

```bash
# א. צור תיקייה חדשה לפרויקט
mkdir ai-product-workflow
cd ai-product-workflow

# ב. העתק את כל הקבצים שהורדת לתיקייה הזו
# (זה תלוי איפה הורדת את הקבצים)

# ג. הפעל את סקריפט ההתקנה
bash setup_project.sh
# (או ב-Windows: sh setup_project.sh)
```

זה יצור את כל מבנה התיקיות:
```
ai-product-workflow/
├── data/raw/
├── data/processed/
├── crews/analyst_crew/
├── crews/scientist_crew/
├── artifacts/analyst/
├── artifacts/scientist/
├── src/
├── tests/
└── notebooks/
```

---

### צעד 2: הקמת Git Repository (3 דקות)

```bash
# א. אתחול Git
git init

# ב. העתק את .gitignore למקום הנכון
cp .gitignore .

# ג. First commit
git add .
git commit -m "Initial project setup"

# ד. (אופציונלי) חבר ל-GitHub
# 1. צור repository חדש ב-GitHub
# 2. הרץ:
git remote add origin https://github.com/<username>/<repo>.git
git branch -M main
git push -u origin main
```

---

### צעד 3: התקנת Python וסביבה וירטואלית (5 דקות)

```bash
# א. בדוק גרסת Python (צריך 3.10+)
python --version
# או
python3 --version

# ב. צור סביבה וירטואלית
python -m venv venv

# ג. הפעל את הסביבה
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# ד. אמת שהסביבה פעילה (אמור לראות (venv) בשורת הפקודה)
which python  # Mac/Linux
where python  # Windows
```

---

### צעד 4: התקנת חבילות Python (10 דקות)

```bash
# א. שדרג pip
pip install --upgrade pip

# ב. התקן את כל החבילות (זה ייקח כמה דקות)
pip install -r requirements.txt

# ג. אמת התקנה מוצלחת
pip list | grep crewai
pip list | grep streamlit
pip list | grep sklearn

# אם הכל עבד, אמור לראות:
# crewai          0.51.0
# streamlit       1.31.0
# scikit-learn    1.4.0
```

**אם יש בעיות בהתקנה:**
```bash
# אם crewai נכשל, נסה:
pip install crewai --no-cache-dir

# אם יש בעיה עם numpy/scipy:
pip install numpy scipy --upgrade
```

---

### צעד 5: הורדת Dataset (5 דקות)

יש לך שתי אפשרויות:

#### אפשרות A: דרך Kaggle CLI (מומלץ)

```bash
# א. התקן Kaggle CLI
pip install kaggle

# ב. הגדר API credentials
# 1. לך ל-https://www.kaggle.com/account
# 2. גלול ל-API section
# 3. לחץ "Create New API Token"
# 4. העבר את kaggle.json למקום הנכון:

# Mac/Linux:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
# העתק את kaggle.json ל: C:\Users\<YourUsername>\.kaggle\

# ג. הרץ את סקריפט ההורדה
python download_dataset.py
# עקוב אחרי ההנחיות והקלד 'y' כשהוא שואל
```

#### אפשרות B: הורדה ידנית (אם CLI לא עובד)

1. לך ל: https://www.kaggle.com/blastchar/telco-customer-churn
2. לחץ "Download" (תצטרך להיכנס לחשבון Kaggle)
3. שמור את הקובץ בשם: `data/raw/dataset.csv`

---

### צעד 6: אימות Dataset (2 דקות)

```bash
# הרץ את הסקריפט לבדיקת ה-dataset
python dataset_selector.py data/raw/dataset.csv
```

אמור לראות:
```
✅ Successfully loaded: data/raw/dataset.csv

📊 BASIC INFORMATION
   Rows: 7,043
   Columns: 21
   
✅ REQUIREMENTS CHECK
   ✅ Row count >= 5,000: 7,043 rows
   ✅ Column count >= 10: 21 columns
   ✅ Has missing values: 11 nulls
   ✅ Mix of types: 3 numeric, 16 categorical
   
🎯 POTENTIAL TARGET VARIABLES
   - Churn (Binary Classification)
   
✅ This dataset is EXCELLENT for the project!
```

---

## ✅ Checklist - אמת שסיימת הכל

- [ ] תיקיית פרויקט נוצרה
- [ ] מבנע תיקיות קיים (data/, crews/, artifacts/)
- [ ] Git repository מאותחל
- [ ] Python 3.10+ מותקן
- [ ] Virtual environment נוצר והופעל (רואה `(venv)` בטרמינל)
- [ ] כל החבילות מ-requirements.txt הותקנו בהצלחה
- [ ] Dataset הורד ונמצא ב-`data/raw/dataset.csv`
- [ ] הרצת `dataset_selector.py` בהצלחה

---

## 🎉 מה הלאה?

אם הכל עבד עד עכשיו - מעולה! 🎊

**אנחנו מוכנים לעבור לשלב הבא:**
- **שלב 1**: בניית Data Analyst Crew (יום אחד)

**תגיד לי שסיימת והכל עבד, ונתחיל לבנות את הסוכנים הראשונים! 🚀**

---

## ❓ פתרון בעיות נפוצות

### בעיה: "python: command not found"
```bash
# נסה עם python3 במקום python
python3 --version
python3 -m venv venv
```

### בעיה: "Permission denied" ב-bash script
```bash
# תן הרשאות להרצה
chmod +x setup_project.sh
bash setup_project.sh
```

### בעיה: pip install נכשל
```bash
# שדרג pip ונסה שוב
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### בעיה: venv לא מפעיל
```bash
# Windows - אם PowerShell חסום:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# אז נסה שוב:
venv\Scripts\activate
```

### בעיה: Kaggle API לא עובד
```bash
# בדוק שה-credentials במקום הנכון
# Mac/Linux:
ls -la ~/.kaggle/kaggle.json

# Windows:
dir C:\Users\%USERNAME%\.kaggle\kaggle.json

# אם הקובץ לא קיים - חזור לשלב 5 אפשרות B (הורדה ידנית)
```

---

## 📞 צריך עזרה?

אם נתקעת בשלב כלשהו:
1. ✅ בדוק את פתרון הבעיות למעלה
2. ✅ העתק את הודעת השגיאה המדויקת
3. ✅ תגיד לי מה לא עובד ואני אעזור!

**אל תתייאש - זה חלק מהתהליך! 💪**
