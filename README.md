# Assignments 1 and 2: Text, Word Embedding and Image Classification API

This FastAPI application extends the Module 3 class activity with a word
embedding endpoint adapted from the Module 2 spaCy practical. It also retains
the original bigram text generation endpoint. Assignment 2 adds a CIFAR-10
image classifier using the EnhancedCNN from Module 4 Practical 3.

## Run with Docker

Install Docker and start its engine (Docker Desktop on macOS). From the project
root, where `Dockerfile`, `pyproject.toml`, and `uv.lock` are located, run:

```bash
docker build -t sps-genai .
docker run --rm --name sps-genai-api -p 8000:80 sps-genai
```

The first build downloads the dependencies and the `en_core_web_lg` model.
The image uses Python 3.13 and installs dependencies from `uv.lock`; no separate
model installation is required. The trained CNN weights in
`models/cifar10_cnn.pt` are copied into the image. Wait for
`Application startup complete`, then
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
| POST | `/classify` | Upload an image and return its predicted CIFAR-10 class. |
| GET | `/docs` | Open Swagger UI to try the endpoints. |

### Image classification (Assignment 2)

In Swagger UI, expand **POST /classify**, choose **Try it out**, upload
`examples/test_ship.png`, and select **Execute**. Or use:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:8000/classify" \
  -F "file=@examples/test_ship.png"
```

The JSON response contains `class_id` (0-9), `class_name`, and `confidence`
(the predicted class's softmax probability). The classes, in order, are:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
This is a classifier for these ten categories, so it always chooses one of
them even when a submitted image belongs to a different category. Confidence
is the model's score, not a guarantee that the prediction is correct.

The included checkpoint achieves **72.93% test accuracy** after 10 epochs.
Two fixed test examples and their actual predictions are recorded in
`examples/predictions.json`. The first example, `test_image.png`, is labeled
cat but this checkpoint predicts dog. This is a model error; successful HTTP
inference does not imply every classification is correct. `test_ship.png`
provides a second example. Retraining can change these results; the generated
metrics and predictions files record each new run.

Images are converted to RGB, resized to 32 by 32, and scaled to [0, 1] with
`ToTensor`, matching the classroom data preprocessing. The API uses evaluation
mode and CPU inference, and reuses the loaded model across requests.

Empty files return HTTP 400, files above 5 MiB or 20 megapixels return 413,
and unreadable images return 422. If the weights are missing, image requests
return 503 with the training command. The Docker build requires the weights.

#### CNN architecture

All convolution layers use a 3 by 3 kernel, stride 1, and padding 1.
Each is followed by BatchNorm, ReLU, and 2 by 2 max pooling with stride 2.

| Stage | Output per image |
| --- | --- |
| RGB input | 3 x 32 x 32 |
| Conv1, BatchNorm, ReLU, MaxPool | 16 x 16 x 16 |
| Conv2, BatchNorm, ReLU, MaxPool | 32 x 8 x 8 |
| Conv3, BatchNorm, ReLU, MaxPool | 64 x 4 x 4 |
| Conv4, BatchNorm, ReLU, MaxPool | 128 x 2 x 2 |
| Flatten | 512 |
| Linear, ReLU, Dropout (0.5) | 128 |
| Linear | 10 logits |

The model has 164,874 trainable parameters. Training uses raw logits with
`CrossEntropyLoss`; softmax is applied only when returning inference scores.

#### Train or retrain

From the project root:

```bash
uv sync --locked --python 3.13
uv run python train_cnn.py
```

The script downloads CIFAR-10 into `data/`, trains on its 50,000 training
images for 10 epochs, and evaluates once on its 10,000 test images. It uses
batch size 32, Adam with learning rate 0.0005, and random seed 42, following
the classroom practical. Apple MPS or CUDA is selected when available,
otherwise training uses the CPU. Device and epoch count can be set explicitly:

```bash
uv run python train_cnn.py --epochs 10 --device cpu
```

The outputs are `models/cifar10_cnn.pt`, `models/training_metrics.json`, and
the sample images and predictions in `examples/`. The metrics file contains the measured accuracy and
training history, including the device and PyTorch version used. Results can
vary across devices. Restart the API or rebuild the Docker image after
retraining so it uses the new weights.

Commit the weights, metrics, example image, source files, and updated
`uv.lock`. The downloaded dataset and local virtual environment are excluded
from Git and the Docker build context. If dependencies are changed, run
`uv lock` and `uv sync --locked` before rebuilding Docker.

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
| `app/cnn_model.py` | Classroom EnhancedCNN architecture and image inference. |
| `train_cnn.py` | CIFAR-10 loading, model training, evaluation, and weight saving. |
| `models/cifar10_cnn.pt` | Trained CNN weights and class order. |
| `models/training_metrics.json` | Actual training history and test results. |
| `examples/` | Two CIFAR-10 test images with their labels and predictions. |
| `pyproject.toml` | Project metadata and declared dependencies, including the spaCy model. |
| `uv.lock` | Resolved dependencies used for installation. |
| `Dockerfile` | Build the image and launch Uvicorn. |
| `.dockerignore` | Exclude the local virtual environment and other unnecessary build files. |
| `.gitignore` | Exclude local environments, caches, and environment variable files from Git. |

The algorithms are adapted from the Module 2 word sampling and word embeddings
practicals and integrated with the Module 3 FastAPI class activity. The CNN
architecture and training settings are adapted from Module 4 Practical 3.
The Linux dependency configuration uses the
[official uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
to install CPU wheels for Docker, while macOS uses PyPI wheels.

## Commit Assignment 2

After checking the Docker endpoint in your existing Git checkout:

```bash
git add app train_cnn.py models examples pyproject.toml uv.lock Dockerfile README.md .gitignore .dockerignore
git commit -m "Add classroom CNN image classification API"
git push
```

Use the repository URL in your assignment submission so the instructor can
clone it and follow the Docker commands above. Include the separate Part 3
calculation answers as required by the assignment page.
