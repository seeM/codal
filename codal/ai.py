import time
from typing import Iterator, Optional

import numpy as np
import openai
from codal.settings import settings


def get_embedding(
    text, model=settings.EMBEDDING_MODEL_NAME, max_attempts=5, retry_delay=1
) -> np.ndarray:
    text = text.replace("\n", " ")
    for attempt in range(max_attempts):
        try:
            raw = openai.Embedding.create(input=[text], model=model)["data"][0][  # type: ignore
                "embedding"
            ]
            array = np.array(raw)
            return array
        # TODO: Only retry on 500 error
        except openai.OpenAIError as exception:
            if attempt < max_attempts - 1:  # No delay on last attempt, TODO: why not?
                time.sleep(retry_delay)
            else:
                raise
    raise Exception("Failed to get embeddings after max attempts")


def get_chat_completion(
    prompt: str, *, model=settings.COMPLETION_MODEL_NAME
) -> Iterator[Optional[str]]:
    for chunk in openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True,
    ):
        content = chunk.choices[0].get("delta", {}).get("content")  # type: ignore
        yield content
