import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def draw_pca(path: str, title: str, ax: Axes):
    data = pd.read_pickle(path)

    emb = np.array(data["embedding"].tolist())

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb)

    human_mask = data["label"] == 1
    nonhuman_mask = data["label"] == 0

    ax.scatter(
        emb_pca[nonhuman_mask, 0],
        emb_pca[nonhuman_mask, 1],
        c="blue",
        label="Non-Human",
        alpha=0.5,
        s=15,
    )

    ax.scatter(
        emb_pca[human_mask, 0],
        emb_pca[human_mask, 1],
        c="red",
        label="Human",
        alpha=0.8,
        s=20,
    )
    ax.set_title(title)
    ax.legend()


def draw_tsne(path: str, title: str, ax: Axes):
    data = pd.read_pickle(path)
    emb = np.array(data["embedding"].tolist())
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        init="random",
        learning_rate="auto",
    )
    emb_tsne = tsne.fit_transform(emb)

    human_mask = data["label"] == 1
    nonhuman_mask = data["label"] == 0

    ax.scatter(
        emb_tsne[nonhuman_mask, 0],
        emb_tsne[nonhuman_mask, 1],
        c="blue",
        label="Non-Human",
        alpha=0.5,
        s=15,
    )

    ax.scatter(
        emb_tsne[human_mask, 0],
        emb_tsne[human_mask, 1],
        c="red",
        label="Human",
        alpha=0.8,
        s=20,
    )

    ax.set_title(title)
    ax.legend()
