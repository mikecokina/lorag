import re
from functools import lru_cache

from typing import Text, List, Dict

import pandas as pd
from tqdm.auto import tqdm
import fitz
from spacy.lang.en import English

from lorag.embeds import embedding


@lru_cache
def get_nlp() -> English:
    return English()


def text_formatter(text: Text) -> Text:
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text


def chunking(text: List[Text], chunk_size: int = 10, overlap: int = 0) -> List[List[Text]]:
    """
    e.g. list of 20 -> list of 10; [20] -> [10, 10] or [25] -> [10, 10, 5]
    """
    chunked = [text[i:i + chunk_size + overlap] for i in range(0, len(text), chunk_size)]

    return chunked


def sentencizer(document: List[Dict]) -> List[Dict]:
    nlp = get_nlp()
    nlp.add_pipe('sentencizer')

    # doc = nlp("I am here. I hate teeth issues? Do you?")
    for item in tqdm(document, desc="Sentencizer"):
        item["sentences"] = list(nlp(item["text"]).sents)

        # Make sure all sentences are strings
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return document


def chunker(document: List[Dict], chunk_size: int = 10, overlap: int = 0) -> List[Dict]:
    for item in tqdm(document, desc="Chunker"):
        item["sentence_chunks"] = chunking(item["sentences"], chunk_size=chunk_size, overlap=overlap)
        item["chunks_count"] = len(item["sentence_chunks"])
    return document


def read_pdf(pdf_file: Text, page_start: int = 1, char_token_length: int = 4) -> List[Dict]:
    document = fitz.open(pdf_file)
    text_ = []

    # noinspection PyTypeChecker
    for page_num, page in tqdm(enumerate(document), total=len(document), desc="Reader"):
        page_num += 1

        text = page.get_text()
        text = text_formatter(text=text)
        text_.append({
            "page_number": page_num - page_start,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text) / char_token_length,
            "text": text
        })

    return text_


def itemize(document: List[Dict]) -> List[Dict]:
    document_ = []
    for item in tqdm(document, desc="Itemizer"):
        for chunk in item["sentence_chunks"]:
            paragraph = "".join(chunk).replace("  ", " ")
            # e.g. .A => . A (capital letter followed by dot)
            paragraph = re.sub(r'\.([A-Z])', r'. \1', paragraph)

            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": paragraph,
                "chunk_char_count": len(paragraph),
                "chunk_word_count": len(paragraph.split(" ")),
                "chunk_token_count": len(paragraph) / 4,
                "chunk_sentence_count": len(chunk),
            }
            document_.append(chunk_dict)
    return document_


def filter_shorts(document: List[Dict], min_size_limit: int = 30) -> List[Dict]:
    df = pd.DataFrame(document)
    df = df[df["chunk_token_count"] > min_size_limit].to_dict(orient="records")
    return df


def embed_paragraphs(document: List[Dict], device: str = 'cpu') -> List[Dict]:
    # Batchin will speed up computing
    text_chunks = [item["sentence_chunk"] for item in document]
    embeddings = embedding(text_chunks, batch_size=32, device=device)

    for item, embed in tqdm(zip(document, embeddings.values()), desc="Embedding enrich", total=len(document)):
        item['embedding'] = embed
    return document


def pdf_preprocessor(pdf_file: Text) -> None:
    path = pdf_file.replace('.pdf', '.csv')
    pdf_data = read_pdf(pdf_file, page_start=42)

    sentencized = sentencizer(pdf_data)
    chunked = chunker(sentencized, overlap=5)
    itemized = itemize(chunked)
    filtered = filter_shorts(itemized, min_size_limit=30)
    emedded = embed_paragraphs(filtered, device="cuda")
    embed_df = pd.DataFrame(emedded)

    # noinspection PyUnresolvedReferences
    embed_df['embedding'] = embed_df['embedding'].apply(lambda x: x.tolist())

    embed_df.to_csv(path, index=False)
