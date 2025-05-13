# Bulletproof: LLM Reasoning Enhancement via Classical Reinforcement Learning

This repository contains the implementation of Bulletproof, a reinforcement learning (RL) framework that enhances the reasoning abilities of open-source language models through Proximal Policy Optimization (PPO).

## Project Overview

Bulletproof addresses a fundamental challenge in large language models (LLMs): robust multi-step reasoning. While recent LLMs have impressive generative capabilities, they still struggle with structured problem-solving, especially on benchmarks like Humanity's Last Exam (HLE).

Our approach uses reinforcement learning to train models to perform reasoning in three structured phases:
- A **thinking** phase where the model breaks down the problem (<think> tokens)
- A **verification** phase to check calculations and logic (<verify> tokens)
- A **conclusion** phase for the final answer (<conclude> tokens)

Models are trained and evaluated on the HLE dataset, focusing on:
- Improving logical consistency
- Reducing hallucinations
- Enhancing answer accuracy

## Repository Structure

```
neuro_240_project/
├── code_files/                 # Main implementation code
│   ├── neuro240/               # Core package
│   │   ├── models/             # Model definitions and setup
│   │   ├── evaluation/         # Evaluation modules 
│   │   ├── training/           # PPO implementation and training logic
│   │   └── utils/              # Utility functions and configuration
│   ├── scripts/                # Executable scripts
│   ├── data/                   # Dataset storage (HLE dataset)
│   ├── outputs/                # Generated artifacts
│   │   ├── models/             # Saved fine-tuned models
│   │   ├── results/            # Evaluation results
│   │   └── plots/              # Generated visualizations
│   ├── setup.py                # Package installation configuration
│   ├── requirements.txt        # Dependencies
│   └── setup_env.sh            # Environment setup script
│
├── final_report/               # Academic paper and presentation materials
│   └── neuro_final_report.tex  # LaTeX source for the paper
│
└── model_performance.csv       # Performance metrics of evaluated models
```

## Installation

1. Clone this repository
2. Navigate to the code_files directory:
   ```bash
   cd code_files
   ```
3. Run the setup script to create a Python 3.9 virtual environment and install dependencies:
   ```bash
   ./setup_env.sh
   ```
4. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```
5. Edit the `.env` file with your API keys:
   - `HF_TOKEN`: Hugging Face API token
   - `OPENAI_API_KEY`: OpenAI API key

## Core Functionality

### Training Models

Train a language model with PPO to improve reasoning:

```bash
cd code_files
python3 scripts/train.py --model_name phi --num_epochs 3
```

Available models:
- TinyLlama (1.1B)
- Phi-2
- StableLM (3B)
- Flan-T5 Small
- GPT-2
- OPT (1.3B)
- Pythia (1.4B)

### Evaluating Models

Evaluate a trained model on the HLE dataset:

```bash
python3 scripts/evaluate.py --model_name phi --compare
```

The `--compare` flag will evaluate both the baseline and fine-tuned versions of the model.

### Analyzing Results

Generate visualizations and performance analysis:

```bash
python3 scripts/analyze_results.py
```

## Methodology

Our approach uses Proximal Policy Optimization (PPO) to train language models with a carefully designed reward function that considers:

1. **Logical consistency** - Evaluating the coherence of reasoning steps
2. **Stepwise correctness** - Assessing the accuracy of each reasoning step
3. **Hallucination penalty** - Detecting and penalizing unsupported claims
4. **Answer accuracy** - Measuring the correctness of the final conclusion

The PPO implementation includes:
- Policy gradient optimization with rewards
- KL divergence regularization to prevent excessive deviation
- Gradient clipping for stability
- Detailed training metrics tracking

## Results

Our evaluations demonstrate that RL-based reasoning token simulation yields measurable improvements in logical coherence and answer accuracy over baseline models. Detailed results are available in the `model_performance.csv` file and in the full paper in the `final_report` directory.

## License

MIT 