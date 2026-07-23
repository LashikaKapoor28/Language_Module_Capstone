import numpy as np
import mygrad as mg
from mygrad.nnet.losses import margin_ranking_loss


def train_val_split(image_ids, train_frac=0.8, seed=0):
    """Split image IDs 4/5 train, 1/5 val (split by image, not by caption,
    so all captions of a validation image stay out of training)."""
    ids = np.array(image_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(train_frac * len(ids))
    return ids[:n_train].tolist(), ids[n_train:].tolist()


def generate_triples(
    image_ids,
    image_to_captions,
    descriptors=None,
    id_to_row=None,
    pool_size=50,
    top_k=5,
    seed=None,
):
    """
    confusors are now chosen by
    semi-hard negative mining: for each image, draw a random pool of
    pool_size other images, rank them by cosine similarity (in raw
    descriptor space) to the true image, and pick the confusor uniformly
    from the top_k most similar ("hardest") candidates in that pool.
    Without descriptors, falls back to a uniformly random confusor.
"""
    rng = np.random.default_rng(seed)
    ids = np.array(image_ids)
    n = len(ids)

    feats = None
    if descriptors is not None:
        rows = np.array([id_to_row[i] for i in image_ids])
        feats = np.asarray(descriptors)[rows].astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8

    triples = []
    for i, img_id in enumerate(image_ids):
        hardest = None
        if feats is not None:
            k_pool = min(pool_size, n - 1)
            pool_idx = rng.choice(n, size=k_pool, replace=False)
            pool_idx = pool_idx[pool_idx != i]
            sims = feats[pool_idx] @ feats[i]
            k = min(top_k, len(pool_idx))
            hardest = pool_idx[np.argpartition(-sims, k - 1)[:k]]

        for cap_id in image_to_captions[img_id]:
            if hardest is not None:
                confusor_id = ids[rng.choice(hardest)]
            else:
                confusor_id = img_id
                while confusor_id == img_id:
                    confusor_id = rng.choice(ids)
            triples.append((cap_id, img_id, confusor_id))
    return triples


def loss_accuracy(w_caption, w_img_true, w_img_confusor, margin=0.25):
    sim_true = mg.einsum("nd,nd->n", w_caption, w_img_true)
    sim_confusor = mg.einsum("nd,nd->n", w_caption, w_img_confusor)
    loss = margin_ranking_loss(sim_true, sim_confusor, 1, margin)
    acc = np.mean(sim_true.data > sim_confusor.data)
    return loss, acc
