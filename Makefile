
PYTHON = uv run python -m src

K ?= 5
MAX_CHUNK_SIZE ?= 2000
METHOD_SEARCH = lexical

QUERY ?=

DATASET_PATH ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
GROUND_TRUTH ?= data/datasets/AnsweredQuestions/dataset_docs_public.json
VLLM_PATH  ?= data/raw/vllm-0.10.1


SEARCH_DIR_SAVE ?= data/output/search_results/UnansweredQuestions/lexical
ANSWER_DIR_SAVE ?= data/output/search_results/AnsweredQuestions/lexical

SEARCH_RESULTS ?= $(SEARCH_DIR_SAVE)/$(notdir $(DATASET_PATH))


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


index_embeddings:
	$(PYTHON) index_embeddings --max_chunk_size $(MAX_CHUNK_SIZE)


search:
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	$(PYTHON) search "$(QUERY)" --k $(K) --method "${METHOD_SEARCH}"


search_dataset:
	$(PYTHON) search_dataset \
		--dataset_path "$(DATASET_PATH)" \
		--k $(K) \
		--save_directory "$(SEARCH_DIR_SAVE)"
		--method semantic


semantic_search:
	$(PYTHON) search_dataset \
		--dataset_path "$(DATASET_PATH)" \
		--k $(K) \
		--save_directory "$(SEARCH_DIR_SAVE)" \
		--method semantic


hybrid_search:
	$(PYTHON)  hybrid_search \
		--dataset_path "$(DATASET_PATH)" \
		--k $(K) \
		--save_directory "$(SEARCH_DIR_SAVE)" \
		--method hybrid


answer:
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	$(PYTHON) answer "$(QUERY)" --k $(K)


answer_dataset:
	$(PYTHON) answer_dataset \
		--student_search_results_path "$(SEARCH_RESULTS)" \
		--save_directory "$(ANSWER_DIR_SAVE)"


incremental_index:
	$(PYTHON) incremental_index  --directory ${VLLM_PATH} --max_chunk_size $(MAX_CHUNK_SIZE)

cache_query:
	@test -n "$(QUERY)" || (echo "Error: QUERY is required" && exit 1)
	$(PYTHON) search "$(QUERY)" --k $(K) --cache True

evaluate:
	$(PYTHON) evaluate \
		--student_search_results_path "$(SEARCH_RESULTS)" \
		--dataset_path "$(GROUND_TRUTH)"


clean:
	rm -rf data/processed
