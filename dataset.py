import torch
from torch.utils.data import Dataset, DataLoader


class ProfileSet(Dataset):
    def __init__(self, id_list, label, text_emb, mean_text_emb, graph_emb=None, mean_graph_emb=None):
        self.id = id_list
        self.label = label
        self.text_emb = text_emb
        self.graph_emb = graph_emb
        self.mean_text_emb = mean_text_emb
        self.mean_graph_emb = mean_graph_emb

    def __getitem__(self, index):
        user = self.id[index]
        if user not in self.text_emb:
            te = self.mean_text_emb
        else:
            te = self.text_emb[user]
        if self.graph_emb:
            if user not in self.graph_emb:
                ge = self.mean_graph_emb
            else:
                ge = self.graph_emb[user]
            return torch.concat((torch.from_numpy(te), torch.from_numpy(ge))), torch.tensor(self.label[user])
        else:
            return torch.from_numpy(te), torch.tensor(self.label[user])

    def __len__(self):
        return len(self.id)

