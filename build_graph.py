import os
import pickle
import csrgraph as cg
from tqdm import tqdm
import networkx as nx
from nodevectors import Node2Vec, GGVec


if not os.path.exists("edge_list.txt"):  # Create edge list

    index = {}

    with open("./data/train/train_links.txt", "r") as f:
        line = f.readline()
        while line:
            u = line[:-1].split(" ")[0]
            if u not in index:
                index[u] = len(index)
            line = f.readline()

    with open("./data/test/test_links.txt", "r") as f:
        line = f.readline()
        while line:
            u = line[:-1].split(" ")[0]
            if u not in index:
                index[u] = len(index)
            line = f.readline()

    with open("other_links.txt", "r") as f:
        line = f.readline()
        while line:
            u = line[:-1].split(" ")[0]
            if u not in index:
                index[u] = len(index)
            line = f.readline()

    with open("index.pkl", "wb") as f:
        pickle.dump(index, f)

    with open("edge_list.txt", "w") as g:
        with open("./data/train/train_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                for b in users[1:]:
                    if b in index:
                        g.write(str(index[a]) + " " + str(index[b]) + "\n")
                line = f.readline()

        with open("./data/test/test_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                for b in users[1:]:
                    if b in index:
                        g.write(str(index[a]) + " " + str(index[b]) + "\n")
                line = f.readline()

        with open("other_links.txt", "r") as f:
            line = f.readline()
            while line:
                users = line[:-1].split(" ")
                a = users[0]
                for b in users[1:]:
                    if b in index:
                        g.write(str(index[a]) + " " + str(index[b]) + "\n")
                line = f.readline()


else:
    with open("index.pkl", "rb") as f:
        index = pickle.load(f)

Graph = cg.read_edgelist("edge_list.txt", directed=True, sep=" ")
ggvec_model = GGVec(n_components=128)
embeddings = ggvec_model.fit_transform(Graph)
print(embeddings)

