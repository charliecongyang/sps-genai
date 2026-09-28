"""Word embeddings adapted from the Module 2 spaCy practical."""

import spacy


class EmbeddingModel:
    """Load the classroom model once and return a word's complete vector."""

    def __init__(self, model_name: str = "en_core_web_lg"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def calculate_embedding(self, input_word: str) -> dict:
        word = input_word.strip()
        if not word:
            raise ValueError("Please enter a non-empty word.")

        # This is the same calculation used in the classroom notebook.
        document = self.nlp(word)
        if len(document) != 1:
            raise ValueError("Please enter one word (one spaCy token).")
        if not document.has_vector:
            raise LookupError(f"No pretrained word vector was found for '{word}'.")

        # NumPy arrays must be converted to a list for the JSON response.
        embedding = document.vector.tolist()
        return {
            "word": word,
            "model": self.model_name,
            "dimension": len(embedding),
            "embedding": embedding,
        }
