import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import evaluate
import torch.nn.functional as F

import matplotlib.pyplot as plt


def adversarial_example(x, y, net, eps=100, lr=1e-2):
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


def get_adv_bulk_data(data, model, eps):
    """
    :param data: a list of tuples of (X,Y) observations
    :param model: any NN model which is supported in this framework
    :param eps: amount by which to perturb. Typically in the range of 0 - 10
    :return: perturbed data
    """
    new_data = []
    for point in data:
        X, Y = point
        adv_X = adversarial_example(X, Y, model, eps)
        new_data.append((adv_X, Y))

    return new_data


def epsilon_sensitivity_trend(models, orig_test_data, num_labels):
    """
    :param num_labels:
    :param models: A dict of models with keys as names and values as trained NN models
    :param orig_test_data: A list tuples of (X,Y) observations
    :return: A dict of timeseries of accuracies varying with epsilon pertrubation
    """

    epsilon_performance = {}
    epsilons = np.arange(0, 6, 0.2)  # increment epsilon in range[0,6] in steps of 0.3

    # Find all examples where models agree on the output label on the raw data
    def models_agree(X, Y, mods):
        all_models_agree = True
        for m in mods:
            pred_Y = m.predict(X)  # Fix the call, depending on which framework is used.
            if pred_Y != Y:
                all_models_agree = False
                break

        return all_models_agree

    new_data = [(X, Y) for X, Y in orig_test_data if models_agree(X, Y, models)]

    for model_name, model in models.items():
        accuracy_trend = []
        for epsilon in epsilons:
            X, Y = get_adv_bulk_data(new_data, model, epsilon)
            acc = evaluate(X, Y, num_labels, model)  # Fix the model vs model.model_forward depending on whats used
            accuracy_trend.append(acc)

        epsilon_performance[model_name] = accuracy_trend

    return epsilon_performance
