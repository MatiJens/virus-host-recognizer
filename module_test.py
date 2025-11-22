import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertModel, BertTokenizer
from peft import LoraConfig, get_peft_model
from pytorch_metric_learning import losses
from Bio import SeqIO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm

# === KONFIGURACJA "TURBO" ===
EPOCHS = 10            # Troszkę mniej epok, bo będziemy mieli więcej danych
TRAIN_LIMIT_NH = 2000  # ZWIĘKSZONE: System RAM to wytrzyma (9.4 -> ~11 GB)

# Parametry techniczne (bezpieczniki)
MAX_LEN = 600          # Zostawiamy (kluczowe dla biologii)
BATCH_SIZE = 16        # Zostawiamy (GPU jest prawie pełne, nie ruszaj tego!)
ACCUMULATION_STEPS = 2 # NOWOŚĆ: Model uczy się jakby miał Batch 32 (lepsza jakość!)
LR = 2e-5

HUMAN_FILE = "human_98.fasta"
NONHUMAN_FILE = "nonhuman_98.fasta"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PRZYGOTOWANIE DANYCH
# ==========================================
class TrainDataset(Dataset):
    def __init__(self, human_path, nonhuman_path, tokenizer, max_len, nh_limit):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        log(f"Wczytywanie danych (Limit Non-Human: {nh_limit})...")
        
        human_seqs = list(SeqIO.parse(human_path, "fasta"))
        for record in human_seqs:
            self.data.append((str(record.seq), 1))
            
        nonhuman_seqs = list(SeqIO.parse(nonhuman_path, "fasta"))
        if len(nonhuman_seqs) > nh_limit:
            random.shuffle(nonhuman_seqs)
            nonhuman_seqs = nonhuman_seqs[:nh_limit]
            
        for record in nonhuman_seqs:
            self.data.append((str(record.seq), 0))
        log(f"Zbiór gotowy: {len(self.data)} sekwencji.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        seq_spaced = " ".join(list(seq))
        inputs = self.tokenizer(seq_spaced, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def get_loader(dataset, batch_size):
    targets = [label for _, label in dataset.data]
    counts = np.bincount(targets)
    weights = 1. / counts
    samples_weights = [weights[l] for l in targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

# ==========================================
# 2. MODEL
# ==========================================
class ContrastiveProtBERT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.bert = base_model
        self.projection = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, 🙂
        proj = F.normalize(self.projection(cls_emb), p=2, dim=1)
        return proj, cls_emb

log("Startujemy...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
base = BertModel.from_pretrained("Rostlab/prot_bert")
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "key", "value"], lora_dropout=0.1, bias="none")
peft_model = get_peft_model(base, peft_config)
model = ContrastiveProtBERT(peft_model).to(device)

dataset = TrainDataset(HUMAN_FILE, NONHUMAN_FILE, tokenizer, MAX_LEN, TRAIN_LIMIT_NH)
loader = get_loader(dataset, BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_func = losses.SupConLoss(temperature=0.1).to(device)
scaler = GradScaler()

# ==========================================
# 3. TRENING Z KUMULACJĄ GRADIENTU
# ==========================================
log(f"START TRENINGU (Gradient Accumulation x{ACCUMULATION_STEPS})")
model.train()
optimizer.zero_grad() # Ważne na start

for epoch in range(EPOCHS):
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoka {epoch+1}/{EPOCHS}", leave=False)
    
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            proj, _ = model(input_ids, mask)
            # Dzielimy stratę przez liczbę kroków kumulacji
            loss = loss_func(proj, labels) / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()

        # Aktualizujemy wagi co ACCUMULATION_STEPS kroków
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Do wyświetlania mnożymy loss z powrotem, żeby widzieć realną wartość
        total_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            
    avg_loss = total_loss / len(loader)
    log(f"Koniec Epoki {epoch+1} | Loss: {avg_loss:.4f}")

log("Trening zakończony. Zapisywanie...")
peft_model.save_pretrained("trained_adapter_turbo")

# ==========================================
# 4. SZYBKA WERYFIKACJA (PCA)
# ==========================================
log("Weryfikacja graficzna...")
eval_seqs, eval_labels = [], []
for r in SeqIO.parse(HUMAN_FILE, "fasta"):
    eval_seqs.append(" ".join(list(str(r.seq)))); eval_labels.append(1)
nh_c = 0
for r in SeqIO.parse(NONHUMAN_FILE, "fasta"):
    eval_seqs.append(" ".join(list(str(r.seq)))); eval_labels.append(0)
    nh_c += 1
    if nh_c >= 300: break # Większa próbka do wykresu

model.eval()
embeddings = []
for i in range(0, len(eval_seqs), 16):
    batch_txt = eval_seqs[i:i+16]
    inputs = tokenizer(batch_txt, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        with autocast():
            _, cls_emb = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            embeddings.append(cls_emb.cpu().float().numpy())

X = np.concatenate(embeddings, axis=0)
y = np.array(eval_labels)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Non-Human', alpha=0.5)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Human', alpha=0.😎
plt.title('PCA - Weryfikacja')
plt.legend(); plt.tight_layout(); plt.savefig("wynik_turbo.png"); plt.show()