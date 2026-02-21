import time

import numpy as np

from .model import forward_batch, sgd_update

TABLE_SIZE = 10_000_000


def build_noise_table(word_counts, power):
    # lookup table where each word occupies slots proportional to count^power
    weighted = np.power(word_counts, power)
    weighted /= weighted.sum()

    table = np.zeros(TABLE_SIZE, dtype=np.int32) # otherwise it would default to float64
    idx = 0
    cumulative = 0.0
    for wid in range(len(word_counts)):
        cumulative += weighted[wid]
        upper = min(int(cumulative * TABLE_SIZE), TABLE_SIZE)
        if upper > idx:
            table[idx:upper] = wid
            idx = upper
    if idx < TABLE_SIZE:
        table[idx:] = len(word_counts) - 1
    return table


def sample_negatives(noise_table, batch_size, num_negatives):
    result = []
    for i in range(batch_size):
        indices = np.random.randint(0, len(noise_table), size=num_negatives)
        result.append(noise_table[indices])
    return np.array(result)


def train(W_in, W_out, pairs, word_counts, num_negatives, batch_size, epochs, lr_start, lr_min, log_every):

    # 0.75 empirically determined in the Google paper
    noise_table = build_noise_table(word_counts, 0.75)

    total_pairs = len(pairs) * epochs
    total_batches = total_pairs // batch_size
    global_step = 0

    print(f"\nTraining: {total_pairs} pairs, {epochs} epoch(s)")
    print(f"LR: {lr_start} -> {lr_min} (linear decay)\n")

    t_start = time.time()

    for epoch in range(epochs):
        shuffled = pairs[np.random.permutation(len(pairs))]

        running_loss = 0.0
        loss_count = 0

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i: i + batch_size]
            centers = batch[:, 0]
            contexts = batch[:, 1]

            # linear lr decay
            progress = global_step / max(total_batches, 1)
            lr = max(lr_start - (lr_start - lr_min) * progress, lr_min)

            negatives = sample_negatives(noise_table, len(batch), num_negatives)

            loss, g_center, g_context, g_neg = forward_batch(
                W_in, W_out, centers, contexts, negatives
            )

            sgd_update(
                W_in, W_out, centers, contexts, negatives,
                g_center, g_context, g_neg, lr
            )

            running_loss += loss
            loss_count += 1
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / loss_count
                elapsed = time.time() - t_start
                speed = (global_step * batch_size) / elapsed
                print(
                    f"  epoch {epoch+1}/{epochs}"
                    f" | step {global_step}/{total_batches}"
                    f" | loss {avg_loss:.4f}"
                    f" | lr {lr:.6f}"
                    f" | {speed:.0f} pairs/s"
                )
                running_loss = 0.0
                loss_count = 0

    elapsed = time.time() - t_start
    print(f"\nTraining finished in {elapsed:.1f}s")
    return W_in
