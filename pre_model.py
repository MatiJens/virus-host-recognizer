from utils.visualisation import draw_pca
import matplotlib.pyplot as plt

normal_protbert_path = "data/processed/protbert.pkl"
contrased_learning_probert_path = "data/processed/contrasted_learning_protbert.pkl"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
draw_pca(normal_protbert_path, "Normal protbert PCA", ax1)
draw_pca(contrased_learning_probert_path, "Contrasted learning protbert PCA", ax2)

plt.tight_layout()
plt.show()
