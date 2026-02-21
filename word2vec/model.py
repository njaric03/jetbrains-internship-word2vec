import numpy as np


def init_embeddings(vocab_size, embed_dim):
    # center embeddings initialized to small random, context embeddings to zeros
    bound = 0.5 / embed_dim
    W_in = np.random.uniform(-bound, bound, (vocab_size, embed_dim))
    W_out = np.zeros((vocab_size, embed_dim))
    return W_in, W_out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_sigmoid(x):
    return np.log(sigmoid(x))


def forward_batch(W_in, W_out, centers, contexts, negatives):
    center_vecs = W_in[centers]
    context_vecs = W_out[contexts]
    neg_vecs = W_out[negatives]

    pos_dot = np.sum(center_vecs * context_vecs, axis=1)
    neg_dot = np.sum(neg_vecs * center_vecs[:, None, :], axis=2)

    # L = -log sigmoid(pos_dot) - sum log sigmoid(-neg_dot)
    batch_size = len(centers)
    loss = (-log_sigmoid(pos_dot).sum() - log_sigmoid(-neg_dot).sum()) / batch_size

    # dL/d(pos_dot) = sigmoid(pos) - 1
    # dL/d(neg_dot) = sigmoid(neg)
    sig_pos = sigmoid(pos_dot)
    sig_neg = sigmoid(neg_dot)

    grad_center = (sig_pos - 1)[:, None] * context_vecs + np.sum(sig_neg[:, :, None] * neg_vecs, axis=1)
    grad_context = (sig_pos - 1)[:, None] * center_vecs
    grad_neg = sig_neg[:, :, None] * center_vecs[:, None, :]

    return loss, grad_center, grad_context, grad_neg


def sgd_update(W_in, W_out, centers, contexts, negatives, grad_center, grad_context, grad_neg, lr):

    for wid in np.unique(centers):
        mask = centers == wid
        W_in[wid] -= lr * grad_center[mask].sum(axis=0)

    for wid in np.unique(contexts):
        mask = contexts == wid
        W_out[wid] -= lr * grad_context[mask].sum(axis=0)

    # negatives have an extra dimension (num_negatives)
    flat_neg = negatives.flatten()
    flat_grad = grad_neg.reshape(-1, grad_neg.shape[-1])
    for wid in np.unique(flat_neg):
        mask = flat_neg == wid
        W_out[wid] -= lr * flat_grad[mask].sum(axis=0)
