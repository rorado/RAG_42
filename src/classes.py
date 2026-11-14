from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


class InvalidFormatError(Exception):
    pass
