# Assignment 1: Text Generation and Word Embedding API

This FastAPI application extends the Module 3 class activity with a word
embedding endpoint adapted from the Module 2 spaCy practical. It also retains
the original bigram text generation endpoint.

## Run with Docker

Install Docker and start its engine (Docker Desktop on macOS). From the project
root, where `Dockerfile`, `pyproject.toml`, and `uv.lock` are located, run:

```bash
docker build -t sps-genai .
docker run --rm --name sps-genai-api -p 8000:80 sps-genai
```

The first build downloads the dependencies and the `en_core_web_lg` model.
The image uses Python 3.13 and installs dependencies from `uv.lock`; no separate
model installation is required. Wait for `Application startup complete`, then
open [the interactive API documentation](http://127.0.0.1:8000/docs).

The container listens on port 80, mapped to port 8000 on the host. Stop any
development server using port 8000 before starting the container. To stop the
container from another terminal:

```bash
docker stop sps-genai-api
```

## Run locally with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use
Python 3.13. From the project root:

```bash
uv sync --locked --python 3.13
uv run uvicorn app.main:app --reload
```

The local server uses the same URLs as the Docker deployment.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Return `{"Hello": "World"}`. |
| GET | `/embedding?word=apple` | Return the complete word embedding. |
| POST | `/generate` | Generate text using the sample corpus and bigram probabilities. |
| GET | `/docs` | Open Swagger UI to try the endpoints. |

### Word embedding

```bash
curl --fail --silent --show-error "http://127.0.0.1:8000/embedding?word=apple"
```

A successful request returns HTTP 200 and a JSON object with these fields:

| Field | Value for this request |
| --- | --- |
| `word` | `"apple"` |
| `model` | `"en_core_web_lg"` |
| `dimension` | `300` |
| `embedding` | An array containing all 300 floating-point vector values. |

The model is loaded once per application process. Following the classroom
practical, the implementation calculates `nlp(word).vector` and converts the
NumPy array to a list for the JSON response. Input is trimmed and must produce
exactly one spaCy token.

Missing or blank input, input longer than 100 characters, and input producing
multiple tokens return HTTP 422. A token without a pretrained vector returns
HTTP 404. These errors include a JSON `detail` field.

### Bigram text generation

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word":"the","length":20}'
```

The response contains `generated_text`. Generation samples from observed
bigram frequencies, so results can vary. `length` is the maximum number of
words, including the starting word. Generation stops early if the current
word has no observed successor.

## Code organization

| File | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI routes, request/response schemas, and HTTP error handling. |
| `app/embedding_model.py` | spaCy model loading, word validation, and vector extraction. |
| `app/bigram_model.py` | Tokenization, bigram probabilities, and text sampling. |
| `pyproject.toml` | Project metadata and declared dependencies, including the spaCy model. |
| `uv.lock` | Resolved dependencies used for installation. |
| `Dockerfile` | Build the image and launch Uvicorn. |
| `.dockerignore` | Exclude the local virtual environment and other unnecessary build files. |
| `.gitignore` | Exclude local environments, caches, and environment variable files from Git. |

The algorithms are adapted from the Module 2 word sampling and word embeddings
practicals and integrated with the Module 3 FastAPI class activity.
