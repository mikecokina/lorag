from typing import Text, Tuple

import torch

import numpy as np

from lorag import embeds


def semantic_search(
        query: Text,
        embeddings: torch.Tensor,
        devide: str = "cpu",
        top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:

    query_embeddings = [*embeds.embedding([query], device=devide).values()][0]
    query_embeddings = torch.tensor(query_embeddings).to(device=devide)
    embeddings = embeddings.to(device=devide)

    dot_product = torch.matmul(query_embeddings, embeddings.T).to('cpu').numpy()
    scores = torch.topk(torch.tensor(dot_product), k=top_k)

    indices = scores.indices.to('cpu').numpy()
    values = scores.values.to('cpu').numpy()

    return values, indices


def rerank_results():
    # mixbread ai reranker
    pass
