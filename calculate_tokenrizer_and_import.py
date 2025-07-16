import logging
from transformers import AutoTokenizer, AutoConfig

class ContextWindowExceededError(Exception):
    pass

def calculate_max_tokens_for_prompt(prompt: str, desired_limit: int = 500, model_id: str = "./llama3-tokenizer") -> int:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    context_window = config.max_position_embeddings
    logger.info(f"Context window size: {context_window}")

    input_ids = tokenizer(prompt)["input_ids"]
    input_token_count = len(input_ids)
    logger.info(f"Input token count: {input_token_count}")

    max_allowed = context_window - input_token_count

    if max_allowed <= 0:
        msg = "Input tokens exceed or equal the context window size; cannot generate output tokens."
        logger.error(msg)
        raise ContextWindowExceededError(msg)

    max_tokens = min(max_allowed, desired_limit)
    logger.info(f"max_tokens for generation: {max_tokens}")
    return max_tokens


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        max_tokens = calculate_max_tokens_for_prompt(prompt)
        logger.info(f"Calculated max_tokens: {max_tokens}")
        # Proceed to call your API with max_tokens here
    except ContextWindowExceededError as e:
        logger.error(f"Cannot generate output: {e}")
        # Handle the situation (e.g., ask user to shorten prompt)
        logger.info("Please shorten your prompt and try again.")
