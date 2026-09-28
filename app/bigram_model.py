"""Bigram text generation for the Module 3 FastAPI application.

Adapted from simple_tokenizer, analyze_bigrams, and generate_text in
Module_2_Practical_2_Word_Sampling.ipynb. Uses only the Python standard library.
"""

from collections import Counter, defaultdict
import random
import re


def simple_tokenizer(text, frequency_threshold=None):
    """Lowercase text and extract words, with optional frequency filtering."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens

    word_counts = Counter(tokens)
    return [
        token for token in tokens
        if word_counts[token] >= frequency_threshold
    ]


def analyze_bigrams(text, frequency_threshold=None):
    """Compute vocabulary and the notebook's bigram probability estimates."""
    words = simple_tokenizer(text, frequency_threshold)
    bigram_counts = Counter(zip(words[:-1], words[1:]))
    unigram_counts = Counter(words)

    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    return list(unigram_counts.keys()), bigram_probs


def generate_text(bigram_probs, start_word, num_words=20):
    """Sample up to num_words words, including the starting word.

    Generation stops early when the current word has no observed successor.
    An unknown starting word is therefore returned on its own. A blank start
    or a nonpositive word limit returns an empty string.
    """
    current_word = start_word.strip().lower()
    if not current_word or num_words <= 0:
        return ""

    generated_words = [current_word]
    for _ in range(num_words - 1):
        next_words = bigram_probs.get(current_word)
        if not next_words:
            break

        # random.choices normalizes the notebook's estimates as weights.
        current_word = random.choices(
            list(next_words.keys()),
            weights=list(next_words.values()),
        )[0]
        generated_words.append(current_word)

    return " ".join(generated_words)


class BigramModel:
    """Wrap the Module 2 functions in the interface used by app/main.py.

    corpus can be one string or an iterable of strings. Multiple strings are
    joined into one training text. Frequency filtering is disabled by default
    so that the small sample corpus in main.py retains its words.
    """

    def __init__(self, corpus, frequency_threshold=None):
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        self.vocab, self.bigram_probs = analyze_bigrams(
            text, frequency_threshold
        )

    def generate_text(self, start_word, length=20):
        """Return at most length words using the notebook's sampling method."""
        return generate_text(self.bigram_probs, start_word, num_words=length)
