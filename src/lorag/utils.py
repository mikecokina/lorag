import ast
import gc
import io
import textwrap
from typing import Text, List, Tuple

import fitz
import numpy as np
import pandas as pd
import torch
from PIL import Image


def read_embed_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    return df


def df_to_embeddings_matrix(df: pd.DataFrame) -> torch.Tensor:
    embeddings = np.stack(df['embedding'].tolist(), dtype=np.float32, axis=0)
    embeddings = torch.tensor(embeddings)
    return embeddings


def print_wraped(text: List[Text], wrap_length: int = 82):
    for txt in text:
        wrapped_text = textwrap.fill(txt, width=wrap_length)
        print(wrapped_text)
        print("\n")


def get_pdf_page(pdf_file: Text, page: int, page_start: int = 42) -> Image.Image:
    document = fitz.open(pdf_file)
    page = document.load_page(page_start + page)
    pix = page.get_pixmap(dpi=300)
    # Determine image mode based on the alpha channel
    mode = "RGBA" if pix.alpha else "RGB"

    # Convert the pixmap to a PIL image using the raw pixel data
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

    document.close()
    return img


def get_cuda_device_string():
    return "cuda"


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
