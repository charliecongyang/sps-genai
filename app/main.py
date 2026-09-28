"""Module 3 text generation API with Assignment 1 word embeddings."""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel


app = FastAPI(title="Text Generation and Word Embedding API")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimension: int
    embedding: list[float]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding", response_model=EmbeddingResponse)
def get_embedding(
    word: str = Query(
        min_length=1,
        max_length=100,
        description="One word to look up, for example apple.",
    ),
):
    """Return the complete pretrained spaCy embedding for one word."""
    try:
        return embedding_model.calculate_embedding(word)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
