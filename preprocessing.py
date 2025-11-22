from utils.preprocessing import add_metadata, concat_data, parse_fasta_with_groups
from embeddings.protBERT import get_protbert_embeddings


# Model hyperparameters
MAX_SEQ_LEN = 1024
BATCH_SIZE = 4

human_raw_path = "data/raw/human_98.fasta"
nonhuman_raw_path = "data/raw/nonhuman_98.fasta"
human_data_path = "data/processed/human_data.pkl"
nonhuman_data_path = "data/processed/nonhuman_data.pkl"

concated_data = "data/processed/concated_data.pkl"

pre_emb_human, metadata_human = parse_fasta_with_groups(human_raw_path, 1)
pre_emb_nonhuman, metadata_nonhuman = parse_fasta_with_groups(nonhuman_raw_path, 0)

protbert_emb_human = get_protbert_embeddings(pre_emb_human, MAX_SEQ_LEN, BATCH_SIZE)
protbert_emb_nonhuman = get_protbert_embeddings(
    pre_emb_nonhuman, MAX_SEQ_LEN, BATCH_SIZE
)

human_labeled = add_metadata(protbert_emb_human, metadata_human)
nonhuman_labeled = add_metadata(protbert_emb_nonhuman, metadata_nonhuman)

concat_data(human_labeled, nonhuman_labeled, out_path=concated_data)
