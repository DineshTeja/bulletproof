import torch
from openai import OpenAI
from utils import prepare_prompt, compute_embedding_similarity
from models import ExtractedResponse


def generate_text(
    model,
    tokenizer,
    question: str,
    answer_type: str,
    openai_client: OpenAI,
    openai_model: str,
    device: torch.device,
    max_length: int = 150,
):
    """
    Generate a structured response using TinyLlama and parse it with OpenAI.

    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer associated with the model
        question: The question to answer
        answer_type: The type of answer expected (e.g., "exactMatch", "multipleChoice")
        openai_client: Instance of the OpenAI client
        openai_model: The OpenAI model to use for parsing
        device: The compute device to use (CPU or GPU)
        max_length: Maximum number of tokens to generate

    Returns:
        A dictionary containing the extracted reasoning components
    """
    model.eval()
    prompt = prepare_prompt(question)

    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    input_ids = encoded.input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
        )

    tinyllama_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    system_prompt = """You are a structured reasoning assistant. Extract the following:
    - thinking: The reasoning process used to arrive at the answer
    - verification: The steps or checks performed to confirm the answer
    - conclusion: The final answer (only the answer, no explanation)
    """

    response = openai_client.beta.chat.completions.parse(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Extract components from:\n{tinyllama_output}",
            },
        ],
        response_format=ExtractedResponse,
        temperature=0,
        max_tokens=900,
    )

    result = response.choices[0].message.parsed
    result.raw_text = tinyllama_output
    return result.dict()


def compute_reward(
    question: str, generated_output: dict, correct_answer: str, answer_type: str
) -> float:
    """
    Compute a reward score based on the quality of the generated answer.

    Args:
        question: The original question
        generated_output: The structured output from the model
        correct_answer: The ground truth answer
        answer_type: The type of answer (e.g., "exactMatch", "multipleChoice")

    Returns:
        A reward score between -1.0 and 1.5
    """
    extracted_answer = generated_output.get("conclusion", "").strip()
    lambda_consistency = 0.4
    lambda_stepwise = 0.3
    lambda_answer = 0.3

    has_thinking = "thinking" in generated_output and generated_output["thinking"]
    stepwise_score = 0.5 if has_thinking else 0.0

    if answer_type == "exactMatch":
        answer_score = compute_embedding_similarity(extracted_answer, correct_answer)
    elif (
        answer_type == "multipleChoice"
        and extracted_answer.upper() == correct_answer.upper()
    ):
        answer_score = 1.0
    else:
        answer_score = 0.0

    reward = (
        lambda_consistency
        + lambda_stepwise * stepwise_score
        + lambda_answer * answer_score
    )
    print(f"\nQuestion: {question}")
    print(f"Generated Output: {generated_output}")
    print(f"Reward: {reward:.4f}")
    return max(min(reward, 1.5), -1.0)


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    openai_client: OpenAI,
    openai_model: str,
    device: torch.device,
):
    """
    Evaluate a model on a dataset and compute average reward.

    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer associated with the model
        eval_dataset: The dataset to evaluate on
        openai_client: Instance of the OpenAI client
        openai_model: The OpenAI model to use for parsing
        device: The compute device to use

    Returns:
        The average reward across all examples
    """
    model.eval()
    rewards = []
    for item in eval_dataset:
        question = item["question"]
        answer = item["answer"]
        answer_type = item["answer_type"]
        generated_output = generate_text(
            model, tokenizer, question, answer_type, openai_client, openai_model, device
        )
        reward = compute_reward(question, generated_output, answer, answer_type)
        rewards.append(reward)
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward: {avg_reward:.4f}")
    return avg_reward
