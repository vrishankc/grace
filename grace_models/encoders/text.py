"""Text encoders for the notes baseline: TF-IDF and averaged BioWordVec embeddings.

BioWordVec (Zhang et al. 2019) ships as a ~13 GB word2vec binary (BioWordVec_PubMed_MIMICIII_d200.vec.bin),
too big to load into 16 GB of RAM and gensim doesn't import in the torch env. So
`load_word2vec_subset` streams the file once and keeps only the words in our training corpus, then caches
the subset to derived/features/ (it is derived from MIMIC vocabulary: keep it local).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..common import FEATURES_DIR

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class TfidfEncoder:
    name = "tfidf"

    def __init__(self, ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=200_000):
        self.vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df,
                                   max_features=max_features, sublinear_tf=True, tokenizer=tokenize,
                                   lowercase=False, token_pattern=None, dtype=np.float32)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        return self.vec.transform(texts)


def load_word2vec_subset(path: str | Path, vocab: set[str]) -> dict[str, np.ndarray]:
    """Stream a word2vec *binary* file and return vectors only for words in `vocab`."""
    import mmap

    out = {}
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        header_end = mm.find(b"\n")
        n_words, dim = map(int, mm[:header_end].split())
        nbytes = 4 * dim
        pos = header_end + 1
        for i in range(n_words):
            while mm[pos] in (10, 13):  # some writers put a newline after each vector
                pos += 1
            sp = mm.find(b" ", pos)
            w = mm[pos:sp].decode("utf-8", errors="ignore")
            if w in vocab:
                out[w] = np.frombuffer(mm[sp + 1: sp + 1 + nbytes], dtype=np.float32).copy()
            pos = sp + 1 + nbytes
            if i % 2_000_000 == 0 and i:
                print(f"  scanned {i:,}/{n_words:,} words, kept {len(out):,}", flush=True)
    return out


class BioWordVecEncoder:
    """Mean of BioWordVec vectors over the document's in-vocabulary tokens (+ an OOV-fraction feature).
    Vocabulary = tokens seen in the TRAIN texts; unseen words at val/test time are ignored."""
    name = "biowordvec_mean"

    def __init__(self, vectors_path: str | Path, cache_tag: str = "default"):
        self.vectors_path = Path(vectors_path)
        self.cache = FEATURES_DIR / f"biowordvec_subset_{cache_tag}.npz"

    def fit(self, texts):
        vocab = set()
        for t in texts:
            vocab.update(tokenize(t))
        if self.cache.exists():
            z = np.load(self.cache, allow_pickle=False)
            if vocab <= set(z["requested"]):  # cache was built for a superset of this vocabulary
                self.vectors = {w: v for w, v in zip(z["words"], z["vectors"]) if w in vocab}
                return self._finish()
        print(f"Streaming {self.vectors_path.name} for {len(vocab):,} train-vocabulary words ...")
        self.vectors = load_word2vec_subset(self.vectors_path, vocab)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, words=np.array(list(self.vectors)), vectors=np.stack(list(self.vectors.values())),
                 requested=np.array(sorted(vocab)))
        return self._finish()

    def _finish(self):
        self.dim = len(next(iter(self.vectors.values())))
        print(f"BioWordVec: {len(self.vectors):,} vectors, dim {self.dim}")
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), self.dim + 1), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            vs = [self.vectors[w] for w in toks if w in self.vectors]
            if vs:
                X[i, :-1] = np.mean(vs, axis=0)
            X[i, -1] = 1 - len(vs) / max(len(toks), 1)  # OOV fraction
        return X
