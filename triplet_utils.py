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


def generate_triples(image_ids, image_to_captions, seed=None):
    """For every caption of every image in `image_ids`, sample a confusor
    image (any other image in the same split).

    Returns
    -------
    list[tuple[caption_id, true_image_id, confusor_image_id]]
    """
    rng = np.random.default_rng(seed)
    ids = np.array(image_ids)
    triples = []
    for img_id in image_ids:
        for cap_id in image_to_captions[img_id]:
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
