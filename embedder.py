from __future__ import annotations
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Sequence
import numpy as np


punch = re.compile("[{}]".format(re.escape(string.punctuation)))

def strip_punc(text: str):
    return punch.sub("", text)

def tokenize(text: str):
    # lowercase, strip punctuation w/ helper above, split on spac
    return strip_punc(text.lower()).split()


def compute_idfs(captions: Iterable[str]):
    captions = list(captions)  
    n_captions = len(captions)
    
    doc_freq = Counter()
    for cap in captions:

        words_in_this_one = set(tokenize(cap))
        doc_freq.update(words_in_this_one)

    idfs = {}
    for word, count in doc_freq.items():
        idfs[word] = np.log10(n_captions / count)

    return idfs

def embed_text(text: str, glove, idfs: Dict[str, float]) -> np.ndarray:
    dim = glove.vector_size
    embedding = np.zeros(dim, dtype=np.float32)

    for word in tokenize(text):
        idf = idfs.get(word, 0.0)
        if idf == 0.0:

            continue
        if word not in glove:
            continue
        embedding += idf * glove[word]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def embed_captions_batch(texts: Sequence[str], glove, idfs: Dict[str, float]):
    dim = glove.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)

    weighted_cache = {}

    for i, text in enumerate(texts):
        row = out[i]
        for word in tokenize(text):
            if word in weighted_cache:
                row += weighted_cache[word]
                continue

            idf = idfs.get(word, 0.0)
            if idf == 0.0 or word not in glove:
                weighted_cache[word] = np.zeros(dim, dtype=np.float32)
                continue

            weighted = (idf * glove[word]).astype(np.float32)
            weighted_cache[word] = weighted
            row += weighted

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  
    out /= norms
    return out
