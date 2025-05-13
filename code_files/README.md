# Neuro240: Neural Reasoning Enhancement

This project focuses on enhancing structured reasoning in language models through reinforcement learning with PPO (Proximal Policy Optimization).

## Project Overview

The project trains language models to follow a structured reasoning approach with:
- A **thinking** phase where the model breaks down the problem
- A **verification** phase to check calculations and logic
- A **conclusion** phase for the final answer

Models are trained and evaluated on the HLE (Human Level Evaluation) dataset with a focus on improving logical consistency, reducing hallucinations, and improving answer accuracy.

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv env`
3. Activate the virtual environment:
   - Windows: `env\Scripts\activate`
   - Mac/Linux: `source env/bin/activate`
4. Install the package: `pip install -e .`
5. Copy `.env.template` to `.env` and add your API keys:
   - Hugging Face token
   - OpenAI API key

## Project Structure

```
neuro240/
├── src/neuro240/           # Main package code
│   ├── models/             # Model definitions and setup
│   ├── evaluation/         # Evaluation modules 
│   ├── training/           # Training logic with PPO
│   └── utils/              # Utility functions
├── data/                   # Data storage (HLE dataset)
├── scripts/                # Runnable scripts
├── outputs/                # Output artifacts (models, plots)
│
├── requirements.txt        # Package dependencies
├── setup.py                # Setup configuration
└── .env.template           # Environment variable template
```

## Usage

1. **Setup Environment**
   ```
   cp .env.template .env
   # Edit .env with your API keys
   ```

2. **Train a Model**
   ```
   python scripts/train.py --model_name phi --epochs 3
   ```

3. **Evaluate a Model**
   ```
   python scripts/evaluate.py --model_name phi --model_path outputs/models/phi
   ```

4. **Analyze Results**
   ```
   python scripts/analyze_results.py --results_file outputs/results/model_performance.csv
   ```

## Models

Supported models:
- TinyLlama (1.1B)
- Phi-2
- StableLM (3B)
- Flan-T5 Small
- GPT-2
- OPT (1.3B)
- Pythia (1.4B)

## License

MIT 