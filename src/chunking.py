from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    size: int
    path: str
    start: int
    end: int

def chunk_file(content: str, path:str, max_chunk:int = 2000) -> list[Chunk]:
    chunks: list[Chunk] = []

    for start in range(0, len(content), max_chunk):
        end = min(start + max_chunk, len(content))
        chunk = Chunk(
            text=content[start:end],
            size=max_chunk,
            path=path,
            start=start,
            end=end
        )
        chunks.append(chunk)
    return chunks


