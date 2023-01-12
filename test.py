import torch
from models import GLinear

s=torch.load("checkpoints/model")
for y in s:
    print(y, s[y].shape)
