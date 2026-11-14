import fire

from .main import (
    answer,
    answer_dataset,
    evaluate,
    incremental_index,
    index,
    index_embeddings,
    search,
    search_dataset,
)


def main():
    fire.Fire({
        "index": index,
        "index_embeddings": index_embeddings,
        "search": search,
        "search_dataset": search_dataset,
        "search_embeddings": search_dataset,
        "hybrid_search": search_dataset,
        "incremental_index": incremental_index,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
