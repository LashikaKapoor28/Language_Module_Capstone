import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from gensim.models import KeyedVectors
from cogworks_data.language import get_data_path

from coco import COCOData
from database import ImageDatabase
from embedder import compute_idfs, embed_text
from triplet_utils import train_val_split


@st.cache_resource
def load_app():
    with Path(get_data_path("resnet18_features.pkl")).open("rb") as f:
        feats = pickle.load(f)

    coco = COCOData(feats.keys())
    glove = KeyedVectors.load_word2vec_format(
        get_data_path("glove.6B.200d.txt.w2v"), binary=False
    )
    idfs = compute_idfs(coco.caption_text.values())

    img_ids = [i for i in coco.image_ids if coco.image_to_captions[i]]
    _, val_ids = train_val_split(img_ids, seed=0)
    desc = np.vstack([feats[i] for i in val_ids]).astype(np.float32)
    urls = {i: coco.image_url[i] for i in val_ids}

    W = np.load("data/W_embed.npy")
    db = ImageDatabase(val_ids, desc, W, image_urls=urls)
    return glove, idfs, db


st.title("Semantic Image Search")
glove, idfs, db = load_app()

query = st.text_input("Query")
k = st.slider("Top-k", 1, 12, 5)

if st.button("Search") and query.strip():
    emb = embed_text(query, glove, idfs)
    ids = db.query(emb, k=k)
    urls = [db.image_urls[i] for i in ids if db.image_urls.get(i)]
    if urls:
        db.display_images(urls)
    else:
        st.write("no results")
