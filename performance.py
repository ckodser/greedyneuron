import torch
from tqdm import tqdm
from models import *
import datasets
import argparse
import simpresnet
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='MLP', type=str,
                        choices={'MLP', 'CNN', 'CNNWide', "LeNET", "ClassifierCNNShit", "ClassifierCNNDeep",
                                 "resnet-18"})
    parser.add_argument('--model_layers', default='2000,2000,2000,2000', type=str, )
    parser.add_argument('--mode', default='normal', type=str, choices={'greedy', 'normal', 'intel', 'greedyExtraverts'})
    parser.add_argument('--dataset', default='MNIST', type=str,
                        choices={'MNIST', "FashionMNIST", "cifar10", "cifar100"})
    parser.add_argument('--learning_rate', default=0.052, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--number_of_worker', default=1, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--run_name', default='not2', type=str)
    parser.add_argument('--normalize', default='False', type=str)
    parser.add_argument('--extravert_bias', default=0, type=float)
    parser.add_argument('--extravert_mult', default=1 / 2, type=float)
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
    noise_eps, pgd_eps, iters = 2, 0.5, 10
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "mode": args.mode,
        "seed": args.seed,
        "model_type": args.model_type,
        "model_layers": args.model_layers,
        "number_of_worker": args.number_of_worker,
        "noise_eps": noise_eps,
        "pgd_eps": pgd_eps,
        "pgd_iters": iters,
    }
    # start_writer(c_run_name, "tensorboard", config)
    # datasets
    trainDataloader, valDataloader, testDataloader, input_shape = datasets.get_dataloaders(dataset_name, batch_size,
                                                                                           args.seed)

    torch.manual_seed(args.seed)

    if args.model_type == "CNN":
        model = ClassifierCNN(input_shape[0], hidden_layers, 10, mode, args.extravert_mult, args.extravert_bias).to(
            device)
    if args.model_type == "CNNWide":
        model = ClassifierCNNWide(input_shape[0], hidden_layers, 10, mode, args.extravert_mult, args.extravert_bias).to(
            device)
    if args.model_type == "ClassifierCNNShit":
        model = ClassifierCNNShit(input_shape[0], hidden_layers, 10, mode, args.extravert_mult, args.extravert_bias).to(
            device)
    if args.model_type == "ClassifierCNNDeep":
        model = ClassifierCNNDeep(input_shape[0], hidden_layers, 10, mode, args.extravert_mult, args.extravert_bias).to(
            device)
    if args.model_type == "MLP":
        model = ClassifierMLP(input_shape[0] * input_shape[1] * input_shape[2], hidden_layers, 10, mode,
                              args.extravert_mult, args.extravert_bias).to(device)
    if args.model_type == "LeNET":
        model = LeNet(10, mode, input_shape[0], args.extravert_mult, args.extravert_bias).to(device)
    if args.model_type == "resnet-18":
        model = simpresnet.resnet18(num_classes=10, normalize=(args.normalize == "True")).to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    for y in model.state_dict():
        print(y, model.state_dict()[y].shape)

    if "greedy" in mode:
        torch.nn.modules.module.register_module_full_backward_hook(hook)
    print(model)

    x=torch.randn((1024, *input_shape))
    y=torch.randint(low=0,high=9,size=(1024,))
    x, y = x.to(device), y.to(device)



    # Record the start time
    start_time = time.time()

    for step in tqdm(range(10000)):
        # forward pass
        model.zero_grad()
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")