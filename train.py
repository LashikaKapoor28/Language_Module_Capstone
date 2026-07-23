import json
import pickle
import string
from collections import Counter
from pathlib import Path

import numpy as np
import mygrad as mg
from gensim.models import KeyedVectors
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.losses import margin_ranking_loss
from cogworks_data.language import get_data_path

D_IMG, D_EMB = 512, 200
MARGIN = 0.25
LR = 2e-3
MOMENTUM = 0.95
BATCH = 32
EPOCHS = 20
STEPS = 500
VAL_STEPS = 50

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def tokenize(text):
    return text.lower().translate(_PUNCT).split()

# temp data functions, replace when everyone done

def load_raw():
    with Path(get_data_path("captions_train2014.json")).open() as f:
        coco = json.load(f)
    with Path(get_data_path("resnet18_features.pkl")).open("rb") as f:
        feats = pickle.load(f)

    img_to_caps = {i: [] for i in feats}
    cap_text = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in feats:
            continue
        cid = ann["id"]
        img_to_caps[iid].append(cid)
        cap_text[cid] = ann["caption"]

    img_ids = [i for i in feats if img_to_caps[i]]
    return img_ids, img_to_caps, cap_text, feats


def build_idf(cap_text):
    N = len(cap_text)
    df = Counter()
    for text in cap_text.values():
        df.update(set(tokenize(text)))
    return {w: np.log10(N / n) for w, n in df.items()}


def embed_caption(text, glove, idf):
    vec = np.zeros(D_EMB, dtype=np.float32)
    for w in tokenize(text):
        if w in glove and w in idf:
            vec += idf[w] * glove[w]
    n = np.linalg.norm(vec)
    if n:
        vec /= n
    return vec


def prep_data(seed=0):
    print("loading data...")
    img_ids, img_to_caps, cap_text, feats = load_raw()
    glove = KeyedVectors.load_word2vec_format(
        get_data_path("glove.6B.200d.txt.w2v"), binary=False
    )
    idf = build_idf(cap_text)
    print("embedding caption../")
    cap_emb = {cid: embed_caption(t, glove, idf) for cid, t in cap_text.items()}

    rng = np.random.default_rng(seed)
    ids = np.array(img_ids)
    rng.shuffle(ids)
    cut = int(0.8 * len(ids))
    train_ids, val_ids = ids[:cut], ids[cut:]

    id_to_row = {int(i): r for r, i in enumerate(ids)}
    desc = np.vstack([feats[int(i)].reshape(1, D_IMG) for i in ids]).astype(np.float32)

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "img_to_caps": img_to_caps,
        "cap_emb": cap_emb,
        "id_to_row": id_to_row,
        "desc": desc,
        "rng": rng,
    }


def sample_batch(img_ids, data, batch=BATCH):
    rng = data["rng"]
    true = rng.choice(img_ids, size=batch, replace=True)
    conf = rng.choice(img_ids, size=batch, replace=True)
    same = conf == true
    while np.any(same):
        conf[same] = rng.choice(img_ids, size=int(same.sum()), replace=True)
        same = conf == true

    caps = np.empty((batch, D_EMB), dtype=np.float32)
    for i, iid in enumerate(true):
        cid = int(rng.choice(data["img_to_caps"][int(iid)]))
        caps[i] = data["cap_emb"][cid]

    t_rows = [data["id_to_row"][int(i)] for i in true]
    c_rows = [data["id_to_row"][int(i)] for i in conf]
    return caps, data["desc"][t_rows], data["desc"][c_rows]


# training

def init_W():
    return glorot_normal(D_IMG, D_EMB)


def embed_images(desc, W):
    w = desc @ W
    return w / mg.sqrt((w ** 2).sum(axis=1, keepdims=True))


def loss_and_acc(caps, true_d, conf_d, W, margin=MARGIN):
    caps = mg.tensor(caps)
    true_d = mg.tensor(true_d)
    conf_d = mg.tensor(conf_d)

    sim_t = mg.einsum("nd,nd->n", embed_images(true_d, W), caps)
    sim_c = mg.einsum("nd,nd->n", embed_images(conf_d, W), caps)

    loss = margin_ranking_loss(sim_t, sim_c, y=1, margin=margin)
    acc = float(np.mean(sim_t.data > sim_c.data))
    return loss, acc


def sgd(W, vel, lr=LR, momentum=MOMENTUM):
    vel[:] = momentum * vel + W.grad
    W.data -= lr * vel
    W.null_grad()


def run_epoch(img_ids, data, W, vel, steps, train=True):
    losses, accs = [], []
    for _ in range(steps):
        caps, td, cd = sample_batch(img_ids, data)
        loss, acc = loss_and_acc(caps, td, cd, W)
        if train:
            loss.backward()
            sgd(W, vel)
        losses.append(float(loss))
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def train(data=None, epochs=EPOCHS, steps=STEPS, val_steps=VAL_STEPS):
    mg.turn_memory_guarding_off()
    if data is None:
        data = prep_data()

    W = init_W()
    vel = np.zeros_like(W.data)
    best_acc, best_W = -1.0, None

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(data["train_ids"], data, W, vel, steps, train=True)
        va_loss, va_acc = run_epoch(data["val_ids"], data, W, vel, val_steps, train=False)
        print(
            f"epoch {ep:02d}  "
            f"train {tr_loss:.4f}/{tr_acc:.3f}  "
            f"val {va_loss:.4f}/{va_acc:.3f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            best_W = np.array(W.data, copy=True)

    W.data[:] = best_W
    print(f"best val acc: {best_acc:.4f}")
    return W


if __name__ == "__main__":
    W = train()
    np.save("data/W_embed.npy", np.asarray(W.data))
    print("saved data/W_embed.npy")
