from tqdm import tqdm
from models import *
from evals import *
from logwriter import *
import datasets
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='MLP', type=str,
                        choices={'MLP', 'CNN', "LeNET"})
    parser.add_argument('--model_layers', default='2000,2000,2000,2000', type=str,
                        choices={'Linear', 'CNN'})
    parser.add_argument('--mode', default='normal', type=str, choices={'greedy', 'normal', 'intel', 'greedyExtraverts'})
    parser.add_argument('--dataset', default='MNIST', type=str, choices={'MNIST', "FashionMNIST"})
    parser.add_argument('--learning_rate', default=0.052, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--number_of_worker', default=1, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--run_name', default='newLinear', type=str)
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
    c_run_name = f"{args.run_name}_{dataset_name}_MODE{mode}_MODEL_{args.model_type}_{str(hidden_layers)[1:-1]}_BS{batch_size}_LR{lr}"


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
        "pgd_iters": iters
    }
    start_writer(c_run_name, "wandb", config)

    # datasets
    trainDataloader, testDataloader = datasets.get_dataloaders(dataset_name, batch_size)

    if args.model_type == "CNN":
        model = ClassifierCNN(hidden_layers, 10, mode).to(device)
    if args.model_type == "MLP":
        model = ClassifierMLP(hidden_layers, 10, mode).to(device)
    if args.model_type == "LeNET":
        model = LeNet(10, mode).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    for y in model.state_dict():
        print(y, model.state_dict()[y].shape)
    set_to_eval(model)
    normal_eval(model, testDataloader, -1, loss_func)

    if "greedy" in mode:
        torch.nn.modules.module.register_module_full_backward_hook(hook)
    print(model)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs // 2, gamma=0.1)

    epoch = 0
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

                logwriter.log("training_monitor/train_loss", loss, epoch, step, silent=True)
                if step % 40 == 1:
                    print(f"train_loss:{avgloss / 40}  epoch:{epoch}, batch:{step}")

                    avgloss = 0

                if step == 0:
                    print("TRACK MODEL")
                    track_model(model, epoch, step)
        scheduler.step()
        set_to_eval(model)
        normal_eval(model, testDataloader, epoch, loss_func)
        noise_robust_eval(model, testDataloader, epoch, loss_func, noise_eps, dataset_name)

    set_to_eval(model)
    strong_robust_eval(model, testDataloader, loss_func, pgd_eps, iters, dataset_name)
    strong_sparse_eval(model, testDataloader, loss_func)
    torch.save(model.state_dict(), f"checkpoints/{c_run_name}/last.pth")
