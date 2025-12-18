#!/bin/bash

# AI Product Workflow - Project Setup Script
echo "🚀 Setting up AI Product Workflow Project..."

# Create main directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p crews/analyst_crew
mkdir -p crews/scientist_crew
mkdir -p artifacts/analyst
mkdir -p artifacts/scientist
mkdir -p src
mkdir -p tests
mkdir -p notebooks
mkdir -p .streamlit

# Create __init__.py files for Python packages
touch crews/__init__.py
touch crews/analyst_crew/__init__.py
touch crews/scientist_crew/__init__.py
touch src/__init__.py
touch tests/__init__.py

# Create placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep

echo "✅ Directory structure created!"

# Show the structure
echo ""
echo "📁 Project Structure:"
tree -L 3 -a 2>/dev/null || find . -type d | sed 's|[^/]*/|  |g'

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Initialize git: git init"
echo "2. Create virtual environment: python -m venv venv"
echo "3. Activate venv: source venv/bin/activate (Mac/Linux) or venv\\Scripts\\activate (Windows)"
echo "4. Install requirements: pip install -r requirements.txt"
