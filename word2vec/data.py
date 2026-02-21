import os
from collections import Counter

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TEXT_PATH = os.path.join(DATA_DIR, "text8")


def load_tokens(text_path):
    with open(text_path, "r") as f:
        text = f.read()
    return text.strip().split()


def build_vocab(tokens, min_count):
    counts = Counter(tokens)
    filtered = [(w, c) for w, c in counts.items() if c >= min_count]
    vocab_words = sorted(filtered, key=lambda x: x[1], reverse=True)
    word_to_id = {}
    id_to_word = []
    word_counts = []
    for idx, (word, count) in enumerate(vocab_words):
        word_to_id[word] = idx
        id_to_word.append(word)
        word_counts.append(count)

    word_counts = np.array(word_counts)
    print(f"Vocabulary: {len(id_to_word)} words (min_count={min_count})")
    return word_to_id, id_to_word, word_counts


def subsample_tokens(token_ids, word_counts, t):
    freqs = word_counts / word_counts.sum()
    # high chance to drop the most common words like "the", no chance for rare
    keep_prob = np.where(freqs > t, np.sqrt(t / freqs), 1.0)
    mask = np.random.random(len(token_ids)) < keep_prob[token_ids]
    result = token_ids[mask]
    print(f"Subsampling: {len(token_ids)} -> {len(result)} tokens")
    return result


def convert_tokens_to_ids(tokens, word_to_id):
    ids = [word_to_id[t] for t in tokens if t in word_to_id]
    return np.array(ids)


def generate_training_pairs(token_ids, window_size):
    n = len(token_ids)
    pairs = []
    for i in range(n):
        # dynamic window size so the more nearby context words appear more often
        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)
        center = token_ids[i]
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))
    pairs = np.array(pairs)
    print(f"Generated {len(pairs)} training pairs")
    return pairs
