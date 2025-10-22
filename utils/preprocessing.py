from Bio import SeqIO
import pandas as pd


def fasta_to_csv(fasta_path: str, csv_path: str, domain: str) -> None:
    """Open fasta file, create DataFrame and fill with organism name, host domain and protein sequences.
    Also save result as *.csv file."""
    with open(fasta_path, "r") as file:
        csv_file = pd.DataFrame(columns=["virus", "host_domain", "protein"])
        for record in SeqIO.parse(file, "fasta"):
            csv_file = pd.concat(
                [
                    pd.DataFrame(
                        [[record.id, domain, record.seq]], columns=csv_file.columns
                    ),
                    csv_file,
                ],
                ignore_index=True,
            )
        csv_file.to_csv(csv_path)
