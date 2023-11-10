import torch
import torch.nn as nn

emb = nn.Embedding(4, 5)
x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
out = emb(x)
print(out)
