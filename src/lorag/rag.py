from typing import Text

from lorag import utils, search, llm


def ask(
        query: Text,
        embeddings_path: Text,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False,
        answer_only: bool = False,
        device: str = 'cuda'

):
    df = utils.read_embed_df(embeddings_path)
    embeddings_matrix = utils.df_to_embeddings_matrix(df)

    sem_res = search.semantic_search(
        query=query,
        embeddings=embeddings_matrix,
        devide="cuda"
    )

    pages_and_chunks = df.to_dict(orient="records")

    res_indices = sem_res[1]
    texts = [pages_and_chunks[index]['sentence_chunk'] for index in res_indices]

    # pages = [pages_and_chunks[index]['page_number'] for index in res_indices]
    # utils.print_wraped(texts)
    # document_page = utils.get_pdf_page(PDF_FILE, pages[0])

    augmented_prompt = llm.prompt_augmentation(
        query=query,
        context=texts
    )

    model_, tokenizer_ = llm.get_llm_model(device='cuda')

    # params_ = llm.get_model_num_parameters(model_)
    # size_ = llm.get_model_mem_size(model_)

    result = llm.generate_text(
        text=augmented_prompt,
        model=model_,
        tokenizer=tokenizer_,
        temperature=temperature,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        answer_only=answer_only,
        device=device
    )

    return result
