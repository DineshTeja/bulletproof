# LLM Reasoning Enhancement via Reinforcement Learning

This repository explores how reinforcement learning techniques can enhance reasoning capabilities in base language models through structured reasoning tokens. We investigate applying Proximal Policy Optimization (PPO) to train non-reasoning language models to use explicit reasoning steps (`<think>`, `<verify>`, `<conclude>`) when approaching complex problems from the Humanity's Last Exam (HLE) benchmark. Our approach aims to improve logical consistency, reduce hallucinations, and increase accuracy on reasoning-intensive tasks by incentivizing step-by-step problem-solving through carefully designed reward functions.

## Small-Scale Experiment

The `main.py` file contains a small-scale experiment implementing the proposed approach using TinyLlama-1.1B as the base model. This preliminary implementation tests the feasibility of enforcing structured reasoning through reinforcement learning on a subset of HLE questions. The experiment demonstrates modest improvements in reasoning capabilities and serves as a proof of concept for our broader research goals.

## Environment Setup

This project requires API credentials from Hugging Face and OpenAI. To run the code:

1. Copy `.env.example` to a new file named `.env`:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file with your actual API keys:
   ```
   HF_TOKEN=your_huggingface_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Make sure to activate your virtual environment and install requirements before running:
   ```
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

Note: The `.env` file is included in `.gitignore` to prevent accidentally committing sensitive credentials.
