#!/bin/bash
# Setup script for Neuro240 project

# Exit on error
set -e

echo "Setting up Neuro240 project environment..."

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "env" ]; then
    echo "Virtual environment already exists, skipping creation."
else
    /usr/local/bin/python3.9 -m venv env
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your API keys:"
    echo "  - HF_TOKEN (Hugging Face)"
    echo "  - OPENAI_API_KEY (OpenAI)"
fi

# Set up NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/hle
mkdir -p outputs/models
mkdir -p outputs/results
mkdir -p outputs/plots

echo "SETUP COMPLETE! You can now run the project scripts."