import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train import evaluate
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


# def fgsm_example(x,y,net,eps):
#     inp = nn.Parameter(x)
#     output = net(inp)
#
#     init_pred = torch.argmax(output)
#
#     if init_pred.item() != y.item():
#         return inp

def adversarial_example(x, y, net, eps=100, lr=1e-2, verbose=False,show_grad=False):
    # inp = nn.Parameter(x + torch.randn_like(x) / 50)
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
        #print("o: ",outputs, "target: ", y)
        loss = criterion(outputs, y)
        #         print("output:", outputs.data, y.data)
        if verbose:
            print("Loss/cnt/eps_curr/Y:", loss.item(), cnt, np.linalg.norm(inp.data - original.data),
                  torch.argmax(outputs).item())
        if torch.argmax(outputs).item() == y.item(): break

        loss.backward()
        if show_grad and cnt == 1:
            print(inp.grad)

        if cnt == 100:
            print("too many steps to create adversary. Exiting.")
            break
        optimizer.step()

    return inp


def random_not_true_label(label):
    possible = list(range(0, label)) + list(range(label + 1, 10))
    choice = random.choice(possible)

    return choice


def get_adv_bulk_data(data, model, eps,limit=None):
    """
    :param data: a list of tuples of (X,Y) observations
    :param model: any NN model which is supported in this framework
    :param eps: amount by which to perturb. Typically in the range of 0 - 10
    :return: perturbed data
    """
    new_X = []
    new_Y = []
    ctr=1
    for point in data:
        X, Y = point
        # Choose a random Y which is not the current Y
        rand_Y = random_not_true_label(Y)
        # Make sure input is shaped like torch likes it
        adv_X = adversarial_example(X.reshape(1, -1), torch.tensor([rand_Y], dtype=torch.long), model, eps=eps, lr=1e-2)
        adv_X = adv_X.reshape(-1)  # Get it back to the original input shape
        new_X.append(adv_X.detach().numpy())

        new_Y.append(Y)
        if limit is not None:
            if ctr == limit:
                break

        ctr += 1

    return np.array(new_X), np.array(new_Y)


def epsilon_sensitivity_trend(models, orig_test_data, num_labels,limit=None,verbose=False):
    """
    :param num_labels:
    :param models: A dict of models with keys as names and values as trained NN models
    :param orig_test_data: A list tuples of (X,Y) observations
    :return: A dict of timeseries of accuracies varying with epsilon pertrubation
    """

    epsilon_performance = {}
    epsilons = np.arange(0, 4, 0.5)  # increment epsilon in range[0,6] in steps of 0.3

    # Find all examples where models agree on the output label on the raw data
    def models_agree(X, Y, mods):
        all_models_agree = True
        for _, m in mods.items():
            pred_Y = m.forward_pass(X.reshape(1, -1)).argmax()  # Fix the call, depending on which framework is used.
            if pred_Y.item() != Y:
                all_models_agree = False
                break

        return all_models_agree

    orig_test_data = [(torch.from_numpy(X).double(), Y) for X, Y in orig_test_data]
    new_data = [(X, Y) for X, Y in orig_test_data if models_agree(X, Y, models)]
    print(len(new_data))
    # new_data = orig_test_data
    for model_name, model in models.items():
        accuracy_trend = []
        print("Model: ", model_name)
        for epsilon in epsilons:
            this_model = model
            X, Y = get_adv_bulk_data(new_data, this_model.forward_pass, epsilon,limit=limit)
            acc, _ = evaluate(X, Y, num_labels, this_model.forward_pass,verbose=verbose)
            accuracy_trend.append(acc)
            print("**************** EPS: ", epsilon)
            print("ACC: ", acc)

        epsilon_performance[model_name] = accuracy_trend

    return epsilon_performance, epsilons
