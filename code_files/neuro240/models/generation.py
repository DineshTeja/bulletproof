"""Text generation and structured response extraction."""

import logging
import torch
import json
from typing import Dict, Optional, Any, Union

from transformers import PreTrainedModel, PreTrainedTokenizer
from openai import OpenAI

from neuro240.utils.config import (
    DEVICE,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_KEY,
    DEFAULT_EVALUATION_CONFIG,
)
from neuro240.utils.types import ModelOutput, ExtractedResponse
from neuro240.models.model_setup import prepare_prompt

logger = logging.getLogger(__name__)


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer_type: str,
    max_length: int = 150,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, str]]:
    """Generate text using a language model and extract structured components.

    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Question to answer
        answer_type: Type of answer expected ("exactMatch", "multipleChoice", etc.)
        max_length: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run inference on

    Returns:
        Dictionary with thinking, verification, conclusion, and raw_text
        or None if generation failed
    """
    if device is None:
        device = DEVICE

    try:
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
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
            )

        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract structured components from output
        extracted = extract_components(model_output)
        
        if extracted:
            return extracted.dict()
        else:
            logger.warning("Failed to extract structured components")
            return {
                "thinking": "",
                "verification": "",
                "conclusion": "",
                "raw_text": model_output
            }

    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)[:100]}...")
        # Return None to indicate we should skip this question
        return None


def extract_components(text: str) -> Optional[ExtractedResponse]:
    """Extract thinking, verification, and conclusion components from text.
    
    This function attempts to extract components with a regex-based approach first,
    then falls back to using the OpenAI API if regex extraction fails.
    
    Args:
        text: The raw model output text
        
    Returns:
        ExtractedResponse object or None if extraction fails
    """
    import re
    
    # Initialize components
    thinking = ""
    verification = ""
    conclusion = ""
    
    # Try to extract components with regex first (faster and cheaper)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    verify_match = re.search(r"<verify>(.*?)</verify>", text, re.DOTALL)
    conclude_match = re.search(r"<conclude>(.*?)</conclude>", text, re.DOTALL)
    
    if think_match and verify_match and conclude_match:
        thinking = think_match.group(1).strip()
        verification = verify_match.group(1).strip()
        conclusion = conclude_match.group(1).strip()
        
        return ExtractedResponse(
            thinking=thinking,
            verification=verification,
            conclusion=conclusion,
            raw_text=text
        )
    
    # If regex fails, try OpenAI extraction if API key is available
    if OPENAI_API_KEY:
        try:
            return extract_with_openai(text)
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {str(e)[:100]}...")
            
    # If all extraction methods fail, return None
    return None


def extract_with_openai(text: str) -> ExtractedResponse:
    """Extract structured response components using OpenAI.
    
    Args:
        text: Raw model output text
        
    Returns:
        ExtractedResponse object
        
    Raises:
        Exception: If OpenAI API call fails
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment")
        
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """You are a structured reasoning assistant. Extract the following:
    - thinking: The reasoning process used to arrive at the answer
    - verification: The steps or checks performed to confirm the answer
    - conclusion: The final answer (only the answer, no explanation)
    """

    response = openai_client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract components from:\n{text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    
    # Parse the response content
    content = response.choices[0].message.content
    result = json.loads(content)
    
    # Create ExtractedResponse object
    return ExtractedResponse(
        thinking=result.get("thinking", ""),
        verification=result.get("verification", ""),
        conclusion=result.get("conclusion", ""),
        raw_text=text
    ) 