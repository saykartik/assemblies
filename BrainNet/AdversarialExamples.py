import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

def adversarial_example(x, y, net,  eps=100, lr=1e-2):
#     inp = nn.Parameter(x + torch.randn_like(x) / 5)
    inp = nn.Parameter(x)
    original = x.clone()

    optimizer = optim.Adam([inp], lr=lr)

    criterion = nn.CrossEntropyLoss()

    cnt = 0
    last = inp.data.clone()
    while True:
        cnt += 1

        optimizer.zero_grad()

        # ensure input stays within [0,1] range for MNIST
        inp.data = torch.max(inp.data, torch.tensor([0.01]).expand_as(inp.data).double())
        inp.data = torch.min(inp.data, torch.tensor([.99]).expand_as(inp.data).double())

        if np.linalg.norm(inp.data - original.data) > eps: break

        outputs = net(inp)

        loss = criterion(outputs, y)
#         print("output:", outputs.data, y.data)
#         print("Loss:", loss.item())
        if torch.argmax(outputs).item() == y.item(): break

        loss.backward()
        optimizer.step()

    return inp