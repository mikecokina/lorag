from lorag import preprocessor
from lorag.rag import ask

DO_PREPROCESSING = False
PDF_FILE = "../../data/human-nutrition-text.pdf"
CSV_FILE = "../../data/human-nutrition-text.csv"


if __name__ == '__main__':
    if DO_PREPROCESSING:
        preprocessor.pdf_preprocessor(PDF_FILE)

    query = "What are the macronutrients, and what roles do they play in the human body?"
    result = ask(
        query=query,
        embeddings_path=CSV_FILE,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        answer_only=True,
        device="cuda"
    )
    print(result)
