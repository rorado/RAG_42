
import fire

from .main import (
    index,
    search,
    search_dataset,
    answer,
    answer_dataset,
    evaluate
)


def main():
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
