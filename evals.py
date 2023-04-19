import copy
import math

import torch.nn as nn
import numpy as np
import torch
import logwriter
import wandb

def norm(A):
    return torch.sum(A * A)


def checkOK(name, A, epoch, step=0):
    eps = 1e-15
    C = A * A
    if len(A.shape) > 1:
        B = torch.sum(C, dim=list(range(1, len(A.shape))))
    else:
        B = C
    B = B[B > eps]
    # if "weight" in name:
    #     print(name, math.sqrt(torch.var(B).item()), torch.min(B).item(), torch.max(B).item())
    logwriter.log(f"correctness_of_greedy/{name}", math.sqrt(torch.var(B).item()) / torch.mean(B).item(), epoch, step,
                  silent=True)


# %%
def track_model(model, epoch, step=0):
    for name, param in model.named_parameters():
        try:
            logwriter.log(f"track_model/w_{name}_norm2", norm(param.data), epoch, step, silent=True)
            logwriter.log(f"track_model/grad_{name}_norm2", norm(param.grad), epoch, step, silent=True)

            # score
            score=torch.flatten(torch.sum(param.data*param.data, dim=1))
            data = [[score[i]] for i in range(score.shape[0])]
            table = wandb.Table(data=data, columns=["utility"])
            wandb.log({f"utility_distribution/w_{name}": wandb.plot.histogram(table, "utility",
           title="utility_distribution"), 'epoch': epoch, 'batch': step})
            logwriter.log(f"track_model/w_{name}_norm2", norm(param.data), epoch, step, silent=True)

            checkOK(name, param.grad, epoch, step)
        except:
            pass


#                 print(name)
# %%
def normal_eval_forgetting(model, testDataloaders, epoch, loss_func, device="cuda", log=True):
    acces=[]
    losses=[]
    for task_id in range(len(testDataloaders)):
        testDataloader=testDataloaders[task_id]
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            for step, (x, y) in enumerate(testDataloader):
                # forward pass
                x, y = x.to(device), y.to(device)
                output = model(x)
                cropped_output = output[:, task_id * 2:task_id * 2 + 2]
                loss_c = loss_func(cropped_output, y - task_id * 2)

                cropped_output = torch.argmax(cropped_output, dim=1)
                loss += loss_c.detach().item()
                total += x.shape[0]
                correct += torch.sum(y-task_id*2 == cropped_output).detach().item()

            acc = (correct / total)
            loss = loss / len(testDataloader)
            acces.append(acc)
            losses.append(loss)
            if log:
                logwriter.log(f"performance_eval/test_loss_{task_id}", loss, epoch)
                logwriter.log(f"performance_eval/test_accuracy_{task_id}", acc, epoch)

    acc, loss= np.mean(np.array(acces)), np.mean(np.array(losses))
    logwriter.log(f"performance_eval/test_loss_average", loss, epoch)
    logwriter.log(f"performance_eval/test_accuracy_average", acc, epoch)
    return acc, loss


def normal_eval_forgetting_hard(model, testDataloaders, epoch, loss_func, device="cuda", log=True, name=""):
    acces=[]
    losses=[]
    for task_id in range(len(testDataloaders)):
        testDataloader=testDataloaders[task_id]
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            for step, (x, y) in enumerate(testDataloader):
                # forward pass
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss_c = loss_func(output, y)

                output = torch.argmax(output, dim=1)
                loss += loss_c.detach().item()
                total += x.shape[0]
                correct += torch.sum(y== output).detach().item()

            acc = (correct / total)
            loss = loss / len(testDataloader)
            acces.append(acc)
            losses.append(loss)
            if log:
                logwriter.log(f"performance_eval/{name}_loss_{task_id}", loss, epoch)
                logwriter.log(f"performance_eval/{name}_accuracy_{task_id}", acc, epoch)

    acc, loss= np.mean(np.array(acces)), np.mean(np.array(losses))
    if log:
        logwriter.log(f"performance_eval/{name}_loss_average", loss, epoch)
        logwriter.log(f"performance_eval/{name}_accuracy_average", acc, epoch)
    return acc, loss, acces


def normal_eval(model, testDataloader, epoch, loss_func, device="cuda", log=True, dataset_name="validation"):
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for step, (x, y) in enumerate(testDataloader):
            # forward pass
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss_c = loss_func(output, y)
            output = torch.argmax(output, dim=1)
            loss += loss_c.detach().item()
            total += x.shape[0]
            correct += torch.sum(y == output).detach().item()

        acc = (correct / total)
        loss = loss / len(testDataloader)
        if log:
            logwriter.log(f"performance_eval/{dataset_name}_loss", loss, epoch)
            logwriter.log(f"performance_eval/{dataset_name}_accuracy", acc, epoch)

    return acc, loss


# PGD Attack
# datasets
pgd_mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2859]),
    'cifar10': np.array([0.49139968, 0.48215827 ,0.44653124])
}
pgd_std = {
    'MNIST': 0.3081,
    'FashionMNIST': 0.2859,
    'cifar10': np.array([0.24703233, 0.24348505, 0.26158768])
}


def pgd_attack(model, images, labels, dataset_name, eps=0.3, alpha=2 / 255, iters=40, t=1, device="cuda"):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data
    min_val = min(((0 - pgd_mean[dataset_name][0]) / pgd_std[dataset_name]))
    max_val = max((1 - pgd_mean[dataset_name][0]) / pgd_mean[dataset_name][0])
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images, t)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=min_val, max=max_val).detach_()

    return images


def add_noise(images, eps, dataset_name, device="cuda"):
    min_val = min(((0 - pgd_mean[dataset_name][0]) / pgd_std[dataset_name]))
    max_val = max((1 - pgd_mean[dataset_name][0]) / pgd_std[dataset_name])
    eta = torch.normal(0.0, eps / 2, images.shape, device=device)
    images = torch.clamp(images + eta, min=min_val, max=max_val).detach_()
    return images


def robust_eval(model, testDataloader, epoch, loss_func, eps, iters, dataset_name, t=1, device="cuda"):
    correct = 0
    total = 0
    loss = 0
    for step, (x, y) in enumerate(testDataloader):
        # forward pass
        x, y = x.to(device), y.to(device)
        x = pgd_attack(model, x, y, dataset_name, eps=eps, alpha=(eps / iters) * 2.5, iters=iters, t=t)
        with torch.no_grad():
            output = model(x)
            loss_c = loss_func(output, y)
            output = torch.argmax(output, dim=1)
            loss += loss_c.detach().item()
            total += x.shape[0]
            correct += torch.sum(y == output).detach().item()

    acc = (correct / total)
    loss = loss / len(testDataloader)
    if epoch is not None:
        logwriter.log("performance_eval/robust_test_loss", loss, epoch)
        logwriter.log("performance_eval/robust_accuracy", acc, epoch)
    else:
        return acc, loss


def strong_robust_eval(model, testDataloader, loss_func, eps, iters, dataset_name):
    # print(f"model {model}, test:{testDataloader}, loss:{loss_func}, eps:{eps}, iters:{iters}, dataset:{dataset_name}")
    t = 0.001
    while t < 100:
        t *= 2
        if t != 1:
            acc, loss = robust_eval(model, testDataloader, None, loss_func, eps, iters, dataset_name, t)
            logwriter.log("performance_eval/robust_test_loss_t", loss, t, translate=False)
            logwriter.log("performance_eval/robust_accuracy_t", acc, t, translate=False)


def sparse_tensor(a, p):
    threshold = torch.quantile(torch.abs(a), p, interpolation='linear')
    a[torch.abs(a) < threshold] = 0
    return a


def sparse_eval(model: torch.nn.Module, testDataloader, loss_func, p):
    s = copy.deepcopy(model.state_dict())
    t = copy.deepcopy(model.state_dict())
    for y in t:
        if "weight" in y:
            t[y] = sparse_tensor(t[y], p)
    model.load_state_dict(t)
    acc, loss = normal_eval(model, testDataloader, None, loss_func, log=False)
    model.load_state_dict(s)
    return acc, loss


def strong_sparse_eval(model, testDataloader, loss_func):
    pruning = [0, 0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    for p in pruning:
        acc, loss = sparse_eval(model, testDataloader, loss_func, p)
        logwriter.log("performance_eval/sparse_loss_t", loss, p * 1000, translate=False)
        logwriter.log("performance_eval/sparse_acc_t", acc, p * 1000, translate=False)


def noise_robust_eval(model, testDataloader, epoch, loss_func, eps, dataset_name, device="cuda"):
    correct = 0
    total = 0
    loss = 0
    for step, (x, y) in enumerate(testDataloader):
        # forward pass
        x, y = x.to(device), y.to(device)
        x = add_noise(x, eps=eps, dataset_name=dataset_name)
        with torch.no_grad():
            output = model(x)
            loss_c = loss_func(output, y)
            output = torch.argmax(output, dim=1)
            loss += loss_c.detach().item()
            total += x.shape[0]
            correct += torch.sum(y == output).detach().item()

    acc = (correct / total)
    loss = loss / len(testDataloader)
    logwriter.log("performance_eval/noise robust_test_loss", loss, epoch)
    logwriter.log("performance_eval/noise robust_accuracy", acc, epoch)


def check_cos_similarity(model, testDataloader, loss_func, layer):
    raise NotImplementedError
    testing_model = ClassifierMLP(hidden_states, 10, prevent_stategic_output_decline, prevent_stategic_weight_increase,
                                  self_interest_neurons).to(device)
    testing_model.load_state_dict(model.state_dict())
    testing_optimizer = torch.optim.SGD(params=testing_model.parameters(), lr=0.0001)
    testing_optimizer.zero_grad()
    tx = None
    for x, y in testDataloader:
        x = x.to(device)
        tx = x
        break
    y = testing_model.assumsion_check_forward(tx, layer, 0)
    ty = torch.normal(0.0, 3, y.shape, device=device)
    loss = torch.sum((y * ty))

    loss.backward()
    testing_optimizer.step()
    with torch.no_grad():
        new_y = testing_model.assumsion_check_forward(tx, layer, 0)
        diff = new_y - y
        ty = ty / torch.sum(ty * ty)
        diff = diff / torch.sum(diff * diff)
        cos = torch.sum(ty * diff)
        return cos.cpu().item()


def assumsion_check(model, testDataloader, loss_func):
    raise NotImplementedError
    for i in range(1, 7):
        cos_i = check_cos_similarity(model, testDataloader, loss_func, i)
        print(f"cos_{i}={cos_i}")
        wandb.log({f"cos_{i}": cos_i, 'epoch': epoch, 'batch': step})
