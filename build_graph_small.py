import os
import pickle
import csrgraph as cg
from tqdm import tqdm
import networkx as nx
import numpy as np
from nodevectors import ProNE


if not os.path.exists("edge_list_small.txt"):  # Create edge list

    chosen = {}
    node_count = 0

    with open("./data/train/train_links.txt", "r") as f:
        line = f.readline()
        while line:
            users = line[:-1].split(" ")
            if users[0] not in chosen:
                chosen[users[0]] = len(chosen)
                node_count += 1
            for u in users[1:]:
                if u not in chosen:
                    chosen[u] = len(chosen)
                    node_count += 1
            line = f.readline()

    with open("./data/test/test_links.txt", "r") as f:
        line = f.readline()
        while line:
            users = line[:-1].split(" ")
            if users[0] not in chosen:
                chosen[users[0]] = len(chosen)
                node_count += 1
            for u in users[1:]:
                if u not in chosen:
                    chosen[u] = len(chosen)
                    node_count += 1
            line = f.readline()

    edge_count = 0

    with open("edge_list_small.txt", "w") as g:
        with open("./data/train/train_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                for b in users[1:]:
                    g.write(a + " " + b + "\n")
                    edge_count += 1
                line = f.readline()

        with open("./data/test/test_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                for b in users[1:]:
                    g.write(a + " " + b + "\n")
                    edge_count += 1
                line = f.readline()

        with open("other_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                if a in chosen:
                    for b in users[1:]:
                        if b in chosen:
                            g.write(a + " " + b + "\n")
                            edge_count += 1
                line = f.readline()
    print(node_count, edge_count)

if not os.path.exists("graph_emb.pkl"):
    Graph = cg.read_edgelist("edge_list_small.txt", directed=True, sep=" ")
    prone = ProNE(n_components=128)
    prone.fit_transform(Graph)
    graph_embedding = {str(i): prone.model[i].astype(np.float32) for i in prone.model}
    with open("graph_emb.pkl", "wb") as f:
        pickle.dump(graph_embedding, f)
else:
    with open("graph_emb.pkl", "rb") as f:
        graph_embedding = pickle.load(f)

mean_graph_emb = np.zeros(128, dtype=np.float32)
for i in graph_embedding:
    mean_graph_emb += graph_embedding[i]
mean_graph_emb /= len(graph_embedding)
