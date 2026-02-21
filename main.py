import os

import numpy as np

from word2vec.data import (
    TEXT_PATH,
    build_vocab,
    convert_tokens_to_ids,
    generate_training_pairs,
    load_tokens,
    subsample_tokens,
)
from word2vec.model import init_embeddings
from word2vec.train import train

EMBED_DIM = 100
WINDOW_SIZE = 5
NUM_NEGATIVES = 5
MIN_COUNT = 5
EPOCHS = 5
BATCH_SIZE = 512
LR = 0.025
LR_MIN = 0.0001
LOG_EVERY = 10_000
SUBSAMPLE_T = 1e-5


def main():
    tokens = load_tokens(TEXT_PATH)
    print(f"Total tokens: {len(tokens)}")

    word_to_id, id_to_word, word_counts = build_vocab(tokens, min_count=MIN_COUNT)
    token_ids = convert_tokens_to_ids(tokens, word_to_id)
    print(f"Tokens after vocab filter: {len(token_ids)}")

    token_ids = subsample_tokens(token_ids, word_counts, SUBSAMPLE_T)
    pairs = generate_training_pairs(token_ids, window_size=WINDOW_SIZE)

    W_in, W_out = init_embeddings(len(id_to_word), EMBED_DIM)

    W_in = train(
        W_in, W_out, pairs, word_counts,
        num_negatives=NUM_NEGATIVES,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr_start=LR,
        lr_min=LR_MIN,
        log_every=LOG_EVERY
    )

    os.makedirs("embeddings", exist_ok=True)
    path = f"embeddings/dim{EMBED_DIM}_neg{NUM_NEGATIVES}_ep{EPOCHS}.npy"
    np.save(path, W_in)
    print(f"\nEmbeddings saved to {path}")


if __name__ == "__main__":
    main()
