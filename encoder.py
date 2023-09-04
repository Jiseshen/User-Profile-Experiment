from transformers import AutoModel, AutoTokenizer
import os
import pickle
import numpy as np
import torch


if not os.path.exists("ernie_emb.pkl"):
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-nano-zh")
    model = AutoModel.from_pretrained("nghuyong/ernie-3.0-nano-zh").to("cuda")

    ernie_embedding = {}
    count = {}
    with torch.no_grad():
        with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = line[:-1].split(",")
                user = line[0]
                if user not in ernie_embedding:
                    ernie_embedding[user] = np.zeros(312, dtype=np.float32)
                    count[user] = 0
                text = line[-1].split("\xa0")[0]
                input = tokenizer(text, return_tensors="pt").to("cuda")
                output = model(**input)
                ernie_embedding[user] += output.last_hidden_state[0][0].cpu().numpy()
                count[user] += 1
                line = f.readline()

        with open("./data/test/test_status.txt", "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = line[:-1].split(",")
                user = line[0]
                if user not in ernie_embedding:
                    ernie_embedding[user] = np.zeros(312, dtype=np.float32)
                    count[user] = 0
                text = line[-1].split("\xa0")[0]
                input = tokenizer(text, return_tensors="pt").to("cuda")
                output = model(**input)
                ernie_embedding[user] += output.last_hidden_state[0][0].cpu().numpy()
                count[user] += 1
                line = f.readline()

    for u in ernie_embedding:
        ernie_embedding[u] /= count[u]
        ernie_embedding[u] = ernie_embedding[u].astype(np.float32)

    with open("ernie_emb.pkl", "wb") as f:
        pickle.dump(ernie_embedding, f)

else:
    with open("ernie_emb.pkl", "rb") as f:
        ernie_embedding = pickle.load(f)


