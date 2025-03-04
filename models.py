from pydantic import BaseModel


class ExtractedResponse(BaseModel):
    """
    Model for structuring the reasoning components of an LLM response.

    Attributes:
        thinking: The reasoning process used to arrive at the answer
        verification: The steps or checks performed to confirm the answer
        conclusion: The final answer (only the answer, no explanation)
        raw_text: The original unprocessed text from the model
    """

    thinking: str
    verification: str
    conclusion: str
    raw_text: str
