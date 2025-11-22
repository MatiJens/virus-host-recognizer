from Bio import SeqIO
import pandas as pd
import numpy as np
import re
import os
from typing import List, Tuple


def add_metadata(embedding: np.ndarray, metadata: List[dict]) -> pd.DataFrame:
    """Open embeddings array, add target value (e.g. 1 or 0) and specific virus name"""

    virus_names = [item["virus_group"] for item in metadata]
    labels = [item["label"] for item in metadata]

    data = pd.DataFrame(
        data={
            "embedding": list(embedding),
            "label": labels,
            "virus_group": virus_names,
        }
    )
    return data


def concat_data(*embeddings: np.ndarray, out_path: str) -> None:
    """Concat DataFrames into one with reseted index and save them as *.pkl file"""
    concat_data = pd.concat(embeddings, ignore_index=True)
    folder_dir = os.path.dirname(out_path)
    if folder_dir and not os.path.exists(folder_dir):
        os.makedirs(folder_dir, exist_ok=True)
    concat_data.to_pickle(out_path)


def parse_fasta_with_groups(file_path: str, label: int) -> Tuple[List[str], List[dict]]:
    """Load FASTA and extracts the virus name from the header."""
    if not os.path.exists(file_path):
        print(f"BŁĄD: Nie znaleziono pliku {file_path}")
        return [], []

    sequences = []
    metadata = []

    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        sequence = " ".join(list(sequence))

        header = record.description
        parts = header.split("|")

        if len(parts) >= 2:
            virus_name = parts[-1].strip().strip(".")
        else:
            virus_name = "Unknown"

        sequences.append(sequence)
        metadata.append(
            {
                "header": header,
                "virus_group": virus_name,
                "label": label,
            }
        )

    print(f"Loaded {len(sequences)} sequences from {file_path}.'")
    return sequences, metadata
