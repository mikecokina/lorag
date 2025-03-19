from functools import lru_cache
from typing import Text, List, Dict

from sentence_transformers import SentenceTransformer


@lru_cache
def get_transformer(device: str = 'cpu') -> SentenceTransformer:
    return SentenceTransformer("all-mpnet-base-v2", device=device)


def embedding(text: List[Text], batch_size: int = 1, device: str = 'cpu') -> Dict:
    t = get_transformer(device=device)

    emebeds = t.encode(
        text,
        batch_size=batch_size,
        convert_to_tensor=False
    )

    ziped = dict(zip(text, emebeds))
    return ziped
