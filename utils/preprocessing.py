from Bio import SeqIO
import pandas as pd
import numpy as np


def add_label(emb_path: str, data_path: str, value: int) -> None:
    """Open file with embeddings saved as *.npy file, add target value (e.g. 1 or 0) and save it as *.csv file."""
    embeddings = np.load(emb_path)
    label = value * np.ones(embeddings.shape[0])
    data = pd.DataFrame(data={"embedding": list(embeddings), "label": label})
    data.to_pickle(data_path)


def concat_data(*in_paths: str, out_path: str) -> None:
    """Concat *.csv files into one with reseted index."""
    frames = [pd.read_pickle(path) for path in in_paths]

    concat_data = pd.concat(frames, ignore_index=True)
    concat_data.to_pickle(out_path)
