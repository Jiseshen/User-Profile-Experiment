import os
import numpy as np
import pickle

if not os.path.exists("emoji_emb.pkl"):

    def extract_emoji(sent):
        emojis = []
        i = 0
        while i < len(sent):
            if len(sent[i]) == 1 and 128506 <= ord(sent[i]) < 130000:    # 😋
                emojis.append(sent[i])
            if "[" in sent[i]:
                left = i
                lag = 0
                while i < len(sent) and "]" not in sent[i]:
                    i += 1
                    lag += 1
                if i < len(sent) and lag < 4:
                    content = "".join(sent[left:i+1])
                    if "http" not in content and "@" not in content:
                        emojis.append(content)
            i += 1
        return emojis

    emoji_map = {}
    emoji_count = {}

    with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line[:-1].split(",")
            text = line[-1].split("\xa0")[0].split(" ")
            for e in extract_emoji(text):
                if e not in emoji_map:
                    emoji_map[e] = len(emoji_map)
                    emoji_count[e] = 0
                emoji_count[e] += 1
            line = f.readline()

    chosen = {}

    for i in emoji_count:
        if emoji_count[i] >= 5:
            chosen[i] = len(chosen)

    print(len(chosen),chosen)

    emoji_embedding = {}

    with open("./data/train/train_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line[:-1].split(",")
            user = line[0]
            if user not in emoji_embedding:
                emoji_embedding[user] = np.zeros(len(chosen))
            text = line[-1].split("\xa0")[0].split(" ")
            for e in extract_emoji(text):
                if e in chosen:
                    emoji_embedding[user][chosen[e]] += 1
            line = f.readline()

    with open("./data/test/test_status.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line[:-1].split(",")
            user = line[0]
            if user not in emoji_embedding:
                emoji_embedding[user] = np.zeros(len(chosen))
            text = line[-1].split("\xa0")[0].split(" ")
            for e in extract_emoji(text):
                if e in chosen:
                    emoji_embedding[user][chosen[e]] += 1
            line = f.readline()

    for i in emoji_embedding:
        emoji_embedding[i] = emoji_embedding[i].astype(np.float32)

    with open("emoji_emb.pkl", "wb") as f:
        pickle.dump(emoji_embedding, f)

else:
    with open("emoji_emb.pkl", "rb") as f:
        emoji_embedding = pickle.load(f)