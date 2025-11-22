import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from peft import LoraConfig, get_peft_model
from pytorch_metric_learning import losses
from typing import List, Optional
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
from peft import PeftModel

HUMAN_FASTA_PATH = "/content/fasta/human_98.fasta"
NONHUMAN_FASTA_PATH = "/content/fasta/nonhuman_98.fasta"


def get_protbert_embeddings(
    sequences_list: List[str],
    max_seq_len: int,
    batch_size: int,
    adapter_path: Optional[str] = None,
) -> np.ndarray:
    """Converts preprocessed samples into embeddings.
    The batch_size parameter speeds up embedding generation, but increases the load on the computer.
    """
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    base_model = BertModel.from_pretrained(model_name)
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

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


def contasted_learning(
    data: List[str],
    targets: List[int],
    max_len: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    accumulation_steps: int,
    output_path: str,
) -> peft.peft_model.PeftModel:
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    encoded_inputs = tokenizer(
        data,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    # Tworzenie Dataset i Loadera
    train_dataset = TensorDataset(
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
        torch.tensor(targets, dtype=torch.long),
    )

    # Sampler do ważenia klas
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    samples_weights = [weights[l] for l in targets]
    sampler = WeightedRandomSampler(
        weights=samples_weights, num_samples=len(data), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = BertModel.from_pretrained("Rostlab/prot_bert")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(base_model, peft_config).to(device)

    projection_head = nn.Sequential(
        nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128)
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(projection_head.parameters()),
        lr=learning_rate,
    )
    loss_func = losses.SupConLoss(temperature=0.1).to(device)
    scaler = GradScaler()

    peft_model.train()
    projection_head.train()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoka {epoch + 1}/{epochs}", leave=False)

        for step, (input_ids, mask, labels) in enumerate(pbar):
            input_ids, mask, labels = (
                input_ids.to(device),
                mask.to(device),
                labels.to(device),
            )

            with autocast():
                cls_emb = peft_model(input_ids, mask).last_hidden_state[:, 0, :]
                proj = F.normalize(projection_head(cls_emb), dim=1)
                loss = loss_func(proj, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        print(f"End of {epoch + 1} epoch | Loss: {total_loss / len(train_loader):.4f}")

    peft_model.save_pretrained(output_path)
    print(f"Model saved to: {output_path}")

    return peft_model
