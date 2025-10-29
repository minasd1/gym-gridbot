#!/bin/bash
# ============================================
#           GridBot Environment Setup
# ============================================

echo "=================================="
echo "Setting up GridBot Environment..."
echo "=================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install the custom gym environment as editable package
echo "Installing gridbot_env as editable package..."
pip install -e .

echo "=================================="
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "=================================="