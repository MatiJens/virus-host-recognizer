import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List

HUMAN_FASTA_PATH = "/content/fasta/human_98.fasta"
NONHUMAN_FASTA_PATH = "/content/fasta/nonhuman_98.fasta"


def get_protbert_embeddings(
    sequences_list: List[str], max_seq_len: int, batch_size: int
) -> np.ndarray:
    """Converts preprocessed samples into embeddings.
    The batch_size parameter speeds up embedding generation, but increases the load on the computer.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "Rostlab/prot_bert_bfd", do_lower_case=False
    )
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(sequences_list), batch_size):
        batch = sequences_list[i : i + batch_size]
        if i % 100 == 0:
            print(f"Batch processing: {i}/{len(sequences_list)}")

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # MEAN POOLING (Kluczowe dla jakości!)
        token_embeddings = outputs.last_hidden_state
        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        all_embeddings.append(mean_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)
