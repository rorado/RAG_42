
import fire

from .main import (
    index,
    search,
    search_dataset,
    answer,
    answer_dataset,
)


def main():
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
    })


if __name__ == "__main__":
    main()
