import numpy as np
import pickle
import os

if not os.path.exists("text_emb.pkl"):
    vocab = {}
    w2v = []

    """
    Word2Vec embedding build
    """

    with open("sgns.weibo.bigram-char", "r", encoding="utf-8") as f:
        _ = f.readline()
        line = f.readline()
        while line:
            line = line[:-2].split(" ")
            vocab[line[0]] = len(vocab)
            w2v.append([float(line[i]) for i in range(1, len(line))])
            line = f.readline()
    w2v = np.array(w2v, dtype=np.float32)

    """
    User Text Feature
    """

    doc_total = 0
    doc_count = np.zeros(len(vocab), dtype=int)

    with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            text = line[:-1].split(",")[-1]
            flag = np.zeros(len(vocab), dtype=int)
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    flag[vocab[i]] = 1
            doc_count += flag
            doc_total += 1
            line = f.readline()

    with open("./data/test/test_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            text = line[:-1].split(",")[-1]
            flag = np.zeros(len(vocab), dtype=int)
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    flag[vocab[i]] = 1
            doc_count += flag
            doc_total += 1
            line = f.readline()

    idf = np.log10(doc_total / (doc_count + 1))

    count_user = {}

    with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            user = line[:-1].split(",")[0]
            if user not in count_user:
                count_user[user] = 0
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    count_user[user] += 1
            line = f.readline()

    with open("./data/test/test_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            user = line[:-1].split(",")[0]
            if user not in count_user:
                count_user[user] = 0
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    count_user[user] += 1
            line = f.readline()

    text_embedding = {u: np.zeros(300) for u in count_user}

    with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line[:-1].split(",")
            user = line[0]
            text = line[-1]
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    text_embedding[user] += w2v[vocab[i]] / count_user[user] * idf[vocab[i]]
            line = f.readline()

    with open("./data/test/test_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line[:-1].split(",")
            user = line[0]
            text = line[-1]
            for i in text.split(" "):
                if "\xa0" in i:
                    break
                if i in vocab:
                    text_embedding[user] += w2v[vocab[i]] / count_user[user] * idf[vocab[i]]
            line = f.readline()

    mean_embedding = np.zeros(300, dtype=np.float32)

    for i in text_embedding:
        text_embedding[i] = text_embedding[i].astype(np.float32)
        mean_embedding += text_embedding[i]

    with open("text_emb.pkl", "wb") as f:
        pickle.dump(text_embedding, f)

    mean_embedding /= len(text_embedding)

else:
    with open("text_emb.pkl", "rb") as f:
        text_embedding = pickle.load(f)

    mean_embedding = np.zeros(300, dtype=np.float32)

    for i in text_embedding:
        mean_embedding += text_embedding[i]

    mean_embedding /= len(text_embedding)
