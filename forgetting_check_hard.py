from tqdm import tqdm
from models import *
from evals import *
from logwriter import *
import datasets
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='MLP', type=str,
                        choices={'MLP', "LeNET"})
    parser.add_argument('--model_layers', default='2000,2000,2000,2000', type=str, )
    parser.add_argument('--mode', default='normal', type=str, choices={'greedy', 'normal'})
    parser.add_argument('--dataset', default='MNIST', type=str,
                        choices={'MNIST', "cifar10"})
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--number_of_worker', default=1, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--run_name', default='forgetting', type=str)
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
    c_run_name = f"{args.run_name}_{dataset_name}_MODE{mode}_MODEL_{args.model_type}_BS{batch_size}_LR{lr}_E{epochs}"

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
        "number_of_worker": args.number_of_worker,
        "task": f"{dataset_name}-Split HARD",
        "step_counting": "epoch*taskid/taskid*200"
    }
    # datasets
    trainDataloaders, valDataloaders, testDataloaders, input_shape = datasets.get_dataloaders_forgetting(dataset_name,
                                                                                                         batch_size,
                                                                                                         seed=args.seed)
    start_writer(forgetting=True, epoches=epochs, steps=len(trainDataloaders[0]))

    if args.model_type == "MLP":
        model = ClassifierMLP(input_shape[0] * input_shape[1] * input_shape[2], hidden_layers, 2, mode).to(device)
    if args.model_type == "LeNET":
        model = LeNet(2, mode, input_shape[0]).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    set_to_eval(model)
    normal_eval_forgetting_hard(model, valDataloaders, -1, loss_func, name="val")
    normal_eval_forgetting_hard(model, testDataloaders, -1, loss_func, name="test")
    if "greedy" in mode:
        torch.nn.modules.module.register_module_full_backward_hook(forgetting_hook)

    for task_id in range(len(trainDataloaders)):
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, (epochs + 1) // 2, gamma=0.1)
        trainDataloader = trainDataloaders[task_id]

        epoch = 0
        end_task = False
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

            print(f"train_loss:{avgloss / (batch_size*len(trainDataloader))}  epoch:{epoch}, batch:{step}")
            avgloss = 0

            scheduler.step()
            set_to_eval(model)
        normal_eval_forgetting_hard(model, valDataloaders, epoch, loss_func, name="val")
        normal_eval_forgetting_hard(model, testDataloaders, epoch, loss_func, name="test")

    normal_eval_forgetting_hard(model, testDataloaders, 0, loss_func, name="real_final_test")
    normal_eval_forgetting_hard(model, valDataloaders, 0, loss_func, name="real_final_val")


# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.05 --batch_size 64
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.05 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.05 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.05 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.05 --batch_size 64
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.05 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.05 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.05 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.05 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.05 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.05 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.05 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.05 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.05 --batch_size 512 

# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.01 --batch_size 64
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.01 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.01 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.01 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.01 --batch_size 64
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.01 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.01 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.01 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.01 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.01 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.01 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.01 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.01 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.01 --batch_size 512 

# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.25 --batch_size 64
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.25 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.25 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.25 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.25 --batch_size 64
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.25 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.25 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.25 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.25 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.25 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.25 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.25 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.25 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.25 --batch_size 512 

# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.002 --batch_size 64
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.002 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.002 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.002 --batch_size 64 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.002 --batch_size 64
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.002 --batch_size 64 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.002 --batch_size 128 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.002 --batch_size 256 
# !python forgetting_check_hard.py --mode greedy --num_epochs 5 --learning_rate 0.002 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 1 --learning_rate 0.002 --batch_size 512 
# !python forgetting_check_hard.py --mode greedy --num_epochs 25 --learning_rate 0.002 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 5 --learning_rate 0.002 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 1 --learning_rate 0.002 --batch_size 512 
# !python forgetting_check_hard.py --mode normal --num_epochs 25 --learning_rate 0.002 --batch_size 512
