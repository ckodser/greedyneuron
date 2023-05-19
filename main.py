import torch
from tqdm import tqdm
import logwriter
from models import *
from evals import *
from logwriter import *
import datasets
import argparse
import simpresnet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='MLP', type=str, choices={'MLP', "LeNET"})
    parser.add_argument('--model_layers', default='2000,2000,2000,2000', type=str, )
    parser.add_argument('--mode', default='normal', type=str, choices={'greedy', 'normal'})
    parser.add_argument('--dataset', default='MNIST', type=str,
                        choices={'MNIST' "cifar10"})
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--number_of_worker', default=1, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--run_name', default='classification', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    # configs
    args = get_args()
    lr = args.learning_rate
    epochs = args.num_epochs
    batch_size = args.batch_size
    device = "cuda"
    dataset_name = args.dataset
    hidden_layers = list(map(int, args.model_layers.split(",")))
    mode = args.mode
    c_run_name = f"{args.run_name}_{dataset_name}_MODE{mode}_BS{batch_size}_LR{lr}_E{epochs}"

    print(c_run_name)
    run = args.run_name
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "mode": args.mode,
        "seed": args.seed,
        "model_type": args.model_type,
        "model_layers": args.model_layers,
        "number_of_worker": args.number_of_worker
    }
    start_writer()

    # datasets
    trainDataloader, valDataloader, testDataloader, input_shape = datasets.get_dataloaders(dataset_name, batch_size,
                                                                                           args.seed)

    torch.manual_seed(args.seed)

    if args.model_type == "MLP":
        model = ClassifierMLP(input_shape[0] * input_shape[1] * input_shape[2], hidden_layers, 10, mode).to(device)
    if args.model_type == "LeNET":
        model = LeNet(10, mode, input_shape[0]).to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    for y in model.state_dict():
        print(y, model.state_dict()[y].shape)
    set_to_eval(model)
    normal_eval(model, testDataloader, -1, loss_func)

    if "greedy" in mode:
        torch.nn.modules.module.register_module_full_backward_hook(hook)
    print(model)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, epochs // 2), gamma=0.1)
    set_name(model)
    epoch = 0
    best_model = {}
    for key in model.state_dict():
        best_model[key] = model.state_dict()[key].clone()
    best_acc = 0
    for epoch in range(0, epochs):
        set_to_train(model)
        logwriter.log("training_monitor/learning_rate", scheduler.get_last_lr()[0], epoch)
        avgloss = 0
        with tqdm(total=len(trainDataloader), position=0, leave=False) as pbar:
            for step, (x, y) in enumerate(trainDataloader):
                # forward pass
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                model.zero_grad()
                output = model(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                pbar.update()

                # tracking
                avgloss += loss.detach().item()

        logwriter.log("training_monitor/train_loss", avgloss/len(trainDataloader), epoch, silent=True)
        scheduler.step()
        set_to_eval(model)
        acc, loss = normal_eval(model, valDataloader, epoch, loss_func)
        if acc > best_acc:
            best_model = {}
            for key in model.state_dict():
                best_model[key] = model.state_dict()[key].clone()
            best_acc = acc
    set_to_eval(model)
    acc, loss = normal_eval(model, valDataloader, epoch, loss_func)
    acc, loss = normal_eval(model, testDataloader, epoch, loss_func, dataset_name="test")

    model.load_state_dict(best_model)
    acc, loss = normal_eval(model, testDataloader, epoch, loss_func, dataset_name="best_test")
    logwriter.finsih()
