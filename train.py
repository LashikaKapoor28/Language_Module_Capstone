import pickle
from pathlib import Path
import numpy as np
import mygrad as mg
from gensim.models import KeyedVectors
from mynn.optimizers.sgd import SGD
from cogworks_data.language import get_data_path
from coco import COCOData
from model import Model
from triplet_utils import train_val_split, generate_triples, loss_accuracy
from embedder import compute_idfs, embed_captions_batch

margin = 0.25
lr = 0.002
momentum = 0.95
batch = 32
epochs = 30
steps = 700
val_steps = 50


def prep_data(seed=0):
    print("loading data...")
    with Path(get_data_path("resnet18_features.pkl")).open("rb") as f:
        feats = pickle.load(f)

    coco = COCOData(feats.keys())
    img_ids = [i for i in coco.image_ids if coco.image_to_captions[i]]

    glove = KeyedVectors.load_word2vec_format(
        get_data_path("glove.6B.200d.txt.w2v"), binary=False
    )
    idfs = compute_idfs(coco.caption_text.values())

    print("embedding captions...")
    cids = list(coco.caption_text)
    embs = embed_captions_batch([coco.caption_text[c] for c in cids], glove, idfs)
    cap_emb = dict(zip(cids, embs))

    train_ids, val_ids = train_val_split(img_ids, seed=seed)
    id_to_row = {i: r for r, i in enumerate(img_ids)}
    desc = np.vstack([feats[i] for i in img_ids]).astype(np.float32)

    kw = dict(descriptors=desc, id_to_row=id_to_row)
    return {
        "cap_emb": cap_emb,
        "id_to_row": id_to_row,
        "desc": desc,
        "train_triples": generate_triples(
            train_ids, coco.image_to_captions, seed=seed, **kw
        ),
        "val_triples": generate_triples(
            val_ids, coco.image_to_captions, seed=seed + 1, **kw
        ),
    }


def get_batch(triples, data, idxs):
    caps, true_d, conf_d = [], [], []
    for j in idxs:
        cid, tid, fid = triples[j]
        caps.append(data["cap_emb"][cid])
        true_d.append(data["desc"][data["id_to_row"][tid]])
        conf_d.append(data["desc"][data["id_to_row"][fid]])
    return np.stack(caps), np.stack(true_d), np.stack(conf_d)


def run_epoch(triples, data, model, optim, steps, rng, do_train):
    losses, accs = [], []
    for i in range(steps):
        idxs = rng.integers(0, len(triples), size=batch)
        caps, td, cd = get_batch(triples, data, idxs)

        loss, acc = loss_accuracy(
            mg.tensor(caps), model(mg.tensor(td)), model(mg.tensor(cd)), margin=margin
        )
        if do_train:
            loss.backward()
            optim.step()
            for p in model.parameters:
                p.null_grad()

        losses.append(float(loss))
        accs.append(float(acc))
    return np.mean(losses), np.mean(accs)


def train(data=None, seed=0):
    mg.turn_memory_guarding_off()
    if data is None:
        data = prep_data(seed)

    model = Model()
    optim = SGD(model.parameters, learning_rate=lr, momentum=momentum)
    rng = np.random.default_rng(seed)
    W = model.parameters[0]

    best_acc, best_W = -1.0, None
    for ep in range(1, epochs + 1):
        tr = run_epoch(data["train_triples"], data, model, optim, steps, rng, True)
        va = run_epoch(data["val_triples"], data, model, optim, val_steps, rng, False)
        print(f"epoch {ep:02d}  train {tr[0]:.4f}/{tr[1]:.3f}  val {va[0]:.4f}/{va[1]:.3f}")

        if va[1] > best_acc:
            best_acc = va[1]
            best_W = np.array(W.data, copy=True)

    W.data[:] = best_W
    print(f"best val acc: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    model = train()
    np.save("data/W_embed.npy", np.asarray(model.parameters[0].data))
    print("saved data/W_embed.npy")
