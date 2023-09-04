from build_vocab import *
from emoji import *
from build_graph_small import *
from encoder import *
from make_label import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


sex_map = ["f", "m"]
age_map = [">1989", "1980-1989", "<1980"]
area_map = ["ne", "n", "e", "m", "s", "sw", "nw", "o"]


def visualization(embed, label, label_map, perplexity=50):
    tsne = TSNE(perplexity=perplexity)
    X_proj = tsne.fit_transform(np.array([embed[u] for u in embed if u in label]))
    state_label = [label_map[label[u]] for u in embed if u in label]
    data = pd.DataFrame({"x": X_proj[:, 0], "y": X_proj[:, 1], "label": state_label})
    sns.scatterplot(data, x="x", y="y", hue="label", legend='full')
    plt.show()


visualization(graph_embedding, sex_train_label, sex_map)
