from utils.preprocessing import fasta_to_csv
import pandas as pd
import os

data_path = "data/raw/"
domains = ["archaea", "bacteria", "eukaryota"]

for domain in ["archaea", "bacteria", "eukaryota"]:
    fasta_path = os.path.join(data_path, f"{domain}.fasta")
    csv_path = os.path.join(data_path, f"{domain}.csv")
    fasta_to_csv(fasta_path, csv_path, domain)
