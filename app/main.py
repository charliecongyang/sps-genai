"""Text generation, word embeddings, and image classification and generation."""

from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.cnn_model import ImageClassifier
from app.embedding_model import EmbeddingModel
from app.gan_generator import GANImageGenerator


app = FastAPI(title="Text, Embeddings, Image Classification and Generation API")

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


class ImageClassificationResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float


@lru_cache(maxsize=1)
def get_image_classifier():
    # Load once when the first image request arrives.
    return ImageClassifier()


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


@app.post("/classify", response_model=ImageClassificationResponse)
def classify_image(file: UploadFile = File(...)):
    """Upload an image and return one of the ten CIFAR-10 classes."""
    image_bytes = file.file.read(5 * 1024 * 1024 + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The image must be 5 MiB or smaller.")
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            if uploaded_image.width * uploaded_image.height > 20_000_000:
                raise HTTPException(status_code=413, detail="The image exceeds 20 megapixels.")
            image = uploaded_image.convert("RGB")
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail="The image dimensions are too large.") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Please upload a valid image, such as PNG or JPEG.") from error
    try:
        return get_image_classifier().predict(image)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error



@lru_cache(maxsize=1)
def get_gan_generator():
    return GANImageGenerator()


@app.get(
    "/generate-image",
    response_class=Response,
    responses={
        200: {
            "description": "A PNG grid of generated handwritten digits.",
            "content": {"image/png": {}},
        }
    },
)
def generate_image(
    num_images: int = Query(
        default=16,
        ge=1,
        le=64,
        description="Number of digit images to generate.",
    ),
    seed: int | None = Query(
        default=None,
        ge=0,
        le=4294967295,
        description="Optional random seed for repeatable results.",
    ),
):
    """Generate a PNG grid of random handwritten digits."""
    try:
        image_bytes = get_gan_generator().generate(
            num_images=num_images,
            seed=seed,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
