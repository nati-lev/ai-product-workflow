# AI Product Workflow - Final Project

> Multi-agent AI system using CrewAI for end-to-end data analysis and predictive modeling

## 🎯 Project Overview

This project simulates a real-world AI product team workflow with two distinct crews:
- **Data Analyst Crew**: Data cleaning, EDA, and contract creation
- **Data Scientist Crew**: Feature engineering and predictive modeling

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ai-product-workflow
```

2. **Create virtual environment**
```bash
# Create venv
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Add your API keys
echo "OPENAI_API_KEY=your-key-here" >> .env
```

5. **Run the project**
```bash
# Run the complete flow
python main_flow.py

# Launch Streamlit app
streamlit run app_streamlit.py
```

## 📁 Project Structure

```
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned data
├── crews/
│   ├── analyst_crew/     # Data Analyst agents & tasks
│   └── scientist_crew/   # Data Scientist agents & tasks
├── artifacts/
│   ├── analyst/          # EDA reports, contracts
│   └── scientist/        # Models, evaluations
├── src/                  # Utility functions
├── tests/                # Unit tests
├── main_flow.py          # Main CrewAI Flow
├── app_streamlit.py      # Streamlit dashboard
└── requirements.txt      # Python dependencies
```

## 📊 Outputs

The project generates:
- ✅ `clean_data.csv` - Cleaned dataset
- ✅ `eda_report.html` - Interactive EDA report
- ✅ `dataset_contract.json` - Data contract
- ✅ `features.csv` - Engineered features
- ✅ `model.pkl` - Trained ML model
- ✅ `evaluation_report.md` - Model evaluation
- ✅ `model_card.md` - Model documentation

## 🛠️ Tech Stack

- **AI Framework**: CrewAI
- **ML**: scikit-learn, XGBoost
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web**: Streamlit, Flask
- **Version Control**: Git, GitHub

## 👥 Team

- [Your Name] - Project Lead
- [Team Member 2] - Data Analyst Crew
- [Team Member 3] - Data Scientist Crew
- [Team Member 4] - Frontend
- [Team Member 5] - Documentation

## 📝 Current Status

- [ ] Project setup
- [ ] Dataset selection
- [ ] Data Analyst Crew implementation
- [ ] Data Scientist Crew implementation
- [ ] Flow integration
- [ ] UI development
- [ ] Deployment
- [ ] Documentation

## 🔗 Links

- [Project Documentation](docs/)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 📧 Contact

For questions or issues, contact: [your-email@example.com]

## 📄 License

This project is for educational purposes - Final Project Course.

---

**Last Updated**: December 2024
