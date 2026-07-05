"""
Local LLM generation using a small open-source instruct model, run
through Hugging Face `transformers` with a PyTorch backend.

Default model: Qwen2.5-0.5B-Instruct
- Small enough to run on CPU with reasonable latency
- Instruction-tuned, so it follows a system prompt + context well
- Swap MODEL_NAME below for a bigger model (e.g. a 1.5B or 3B variant)
  if you have a GPU available.
"""

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load tokenizer + model once (singleton), reused across all requests."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # float32 for CPU compatibility
        device_map="cpu",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_response(system_prompt: str, context_chunks: list[str], user_query: str,
                       max_new_tokens: int = 256) -> str:
    """
    Build a chat-formatted prompt from the agent's persona + retrieved
    RAG context + the user's question, then generate a reply.
    """
    context_block = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else \
        "(No specific knowledge base entries were found for this query.)"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Relevant knowledge base context:\n{context_block}\n\n"
                f"User question: {user_query}\n\n"
                "Answer using the context above where relevant. If the context "
                "doesn't cover it, answer helpfully using general knowledge and "
                "say so."
            ),
        },
    ]

    generator = _get_pipeline()
    output = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    # transformers chat pipelines return the full conversation; take the
    # last assistant turn as the reply.
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()
