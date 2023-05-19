import numpy as np
import torch
import logwriter


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

