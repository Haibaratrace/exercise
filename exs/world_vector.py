import torch
import torchtext

gv = torchtext.vocab.GloVe(name='6B', dim=50)


def get_vector(word):
    return gv.vectors[gv.stoi[word]]


def sim_10(word, n=10):
    all_distance = [(gv.itos[i], torch.dist(word, w)) for i, w in enumerate(gv.vectors)]
    return sorted(all_distance, key=lambda t: t[1])[:n]


def answer(w1, w2, w3):
    v4 = get_vector(w3) - get_vector(w1) + get_vector(w2)
    return sim_10(v4)[0][0]


print(answer("china", "beijing", "japan"))
