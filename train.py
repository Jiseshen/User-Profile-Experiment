from build_graph_small import *
from build_vocab import *
from emoji import *
from encoder import *
from make_label import *  # data split done there
from mlp import *
from dataset import *


torch.manual_seed(30)

# text_embedding = {i: np.concatenate((text_embedding[i],emoji_embedding[i])) for i in text_embedding}
# text_embedding = {i: np.concatenate((text_embedding[i], ernie_embedding[i])) for i in text_embedding}
# text_embedding = ernie_embedding
text_embedding = emoji_embedding

graph_dim = 128
text_dim = 525

area_train_set = ProfileSet(area_train_uid, area_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
area_train_loader = DataLoader(area_train_set, batch_size=64, shuffle=True)
area_dev_set = ProfileSet(area_dev_uid, area_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
area_dev_loader = DataLoader(area_dev_set, batch_size=len(area_dev_set), shuffle=False)
area_test_set = ProfileSet(test_uid, area_test_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
area_test_loader = DataLoader(area_test_set, batch_size=len(area_test_set), shuffle=False)

age_train_set = ProfileSet(train_uid, age_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
age_train_loader = DataLoader(age_train_set, batch_size=64, shuffle=True)
age_dev_set = ProfileSet(dev_uid, age_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
age_dev_loader = DataLoader(age_dev_set, batch_size=len(age_dev_set), shuffle=False)
age_test_set = ProfileSet(test_uid, age_test_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
age_test_loader = DataLoader(age_test_set, batch_size=len(age_test_set), shuffle=False)

sex_train_set = ProfileSet(train_uid, sex_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
sex_train_loader = DataLoader(sex_train_set, batch_size=64, shuffle=True)
sex_dev_set = ProfileSet(dev_uid, sex_train_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
sex_dev_loader = DataLoader(sex_dev_set, batch_size=len(sex_dev_set), shuffle=False)
sex_test_set = ProfileSet(test_uid, sex_test_label, text_embedding, mean_embedding, graph_embedding, mean_graph_emb)
sex_test_loader = DataLoader(sex_test_set, batch_size=len(sex_test_set), shuffle=False)


def evaluate(model, loader, class_num):
    model.eval()
    with torch.no_grad():
        for feature, label in loader:
            logit = model.forward(feature)
            acc = torch.eq(logit.argmax(dim=1), label).float().mean()
            predClass = torch.eq(logit, logit.max(dim=1, keepdims=True).values).float()
            trueClass = F.one_hot(label, class_num).float()
            TP = (predClass * trueClass).sum(dim=0)
            FP = (predClass * (1 - trueClass)).sum(dim=0)
            FN = ((1 - predClass) * trueClass).sum(dim=0)
            microF1 = (2 * TP.sum()) / (2 * TP.sum() + FP.sum() + FN.sum())
            macroF1 = ((2 * TP) / (2 * TP + FP + FN + 1e-10)).mean()
    return acc, microF1, macroF1


def train(model, train_loader, dev_loader, test_loader, class_num, epoch):
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-2)
    loss = nn.CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        for feature, label in train_loader:
            optimizer.zero_grad()
            logit = model(feature)
            l = loss(logit, label)
            l.backward()
            optimizer.step()
        print("After epoch {}, the train loss is {}".format(i, l.item()))
        acc, _, _ = evaluate(model, dev_loader, class_num)
        print("The dev acc is {}".format(acc))
    acc, microF1, macroF1 = evaluate(model, test_loader, class_num)
    print("The final test acc, micro-F1, macro_F1 are {}, {} and {}".format(acc, microF1, macroF1))



print("Train sex classifier:")
SexMLP = MLP(text_dim+graph_dim, 100, 2)
train(SexMLP, sex_train_loader, sex_dev_loader, sex_test_loader, 2, 5)

# print("Train age classifier:")
# AgeMLP = MLP(text_dim+graph_dim, 100, 3)
# train(AgeMLP, age_train_loader, age_dev_loader, age_test_loader, 3, 5)

# print("Train area classifier:")
# AreaMLP = MLP(text_dim+graph_dim, 100, 8)
# train(AreaMLP, area_train_loader, area_dev_loader, area_test_loader, 8, 20)
