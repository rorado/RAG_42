
PYTHON = uv run python -m src

K ?= 5
MAX_CHUNK_SIZE ?= 2000

QUERY ?=

DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
GROUND_TRUTH ?= data/datasets/AnsweredQuestions/dataset_docs_public.json

SEARCH_DIR_SAVE ?= data/output/search_results/UnansweredQuestions
ANSWER_DIR_SAVE ?= data/output/search_results/AnsweredQuestions

SEARCH_RESULTS ?= $(SEARCH_DIR)/$(notdir $(DATASET_PATH))


.PHONY: help install index search search_dataset answer answer_dataset evaluate clean


help:
	@echo "Available commands:"
	@echo "  make install"
	@echo "  make index"
	@echo "  make search QUERY='your question'"
	@echo "  make search_dataset"
	@echo "  make answer QUERY='your question'"
	@echo "  make answer_dataset"
	@echo "  make evaluate"
	@echo "  make clean"


install:
	uv sync


index:
	$(PYTHON) index --max_chunk_size $(MAX_CHUNK_SIZE)


search:
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	$(PYTHON) search "$(QUERY)" --k $(K)


search_dataset:
	$(PYTHON) search_dataset \
		--dataset_path "$(DATASET_PATH)" \
		--k $(K) \
		--save_directory "$(SEARCH_DIR_SAVE)"


answer:
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	$(PYTHON) answer "$(QUERY)" --k $(K)


answer_dataset:
	$(PYTHON) answer_dataset \
		--student_search_results_path "$(SEARCH_RESULTS)" \
		--save_directory "$(ANSWER_DIR_SAVE)"


evaluate:
	$(PYTHON) evaluate \
		--student_search_results_path "$(SEARCH_RESULTS)" \
		--dataset_path "$(GROUND_TRUTH)"


clean:
	rm -rf data/processed
