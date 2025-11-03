from utils.preprocessing import add_label, concat_data

human_emb_path = "data/raw/emb_human.npy"
human_data_path = "data/processed/human_data.pkl"
nonhuman_emb_path = "data/raw/emb_nonhuman.npy"
nonhuman_data_path = "data/processed/nonhuman_data.pkl"

concated_data = "data/processed/concated_data.pkl"

add_label(human_emb_path, human_data_path, 1)
add_label(nonhuman_emb_path, nonhuman_data_path, 0)
concat_data(human_data_path, nonhuman_data_path, out_path=concated_data)
