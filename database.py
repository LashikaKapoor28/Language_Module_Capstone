import numpy as np
import streamlit as st


class ImageDatabase:
    def __init__(self, image_ids, descriptors, W_embed, image_urls=None):
        self.image_ids = list(image_ids)
        self.W_embed = np.asarray(W_embed)
        self.image_urls = image_urls or {}
        self.embeddings = self.descriptor_to_embedding(np.asarray(descriptors))

    def descriptor_to_embedding(self, descriptor):
        w = descriptor @ self.W_embed
        norm = np.linalg.norm(w, axis=-1, keepdims=True)
        return w / (norm + 1e-8)

    def query(self, caption_embedding, k=10):
        if isinstance(caption_embedding, (list, tuple)):
            caption_embedding = np.array(caption_embedding)
        caption_embedding = np.asarray(caption_embedding)
        caption_embedding = caption_embedding / (np.linalg.norm(caption_embedding) + 1e-8)
        sims = self.embeddings @ caption_embedding
        k = min(k, len(self.image_ids))
        top_k_idx = np.argpartition(-sims, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        return [self.image_ids[i] for i in top_k_idx]

    def display_images(self, urls):
        st.image(urls, width=300)
