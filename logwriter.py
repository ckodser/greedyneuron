from torch.utils.tensorboard import SummaryWriter
import datetime
import wandb

# Writer will output to ./runs/ directory by default
writer = None
max_step = 1
writer_mode = None
disable = False
previous_epoch = -1
task_id = 0
forgetting_setting = False
total_epoches = None


def start_writer(name, mode="tensorboard", dict=None, forgetting=False, epoches=None, steps=None):
    if disable:
        return
    global writer, writer_mode, forgetting_setting, total_epoches, max_step
    total_epoches = epoches
    max_step= steps
    forgetting_setting = forgetting
    writer_mode = mode
    if mode == "tensorboard":
        writer = SummaryWriter("runs/" + name + "_" + str(datetime.datetime.now()).replace(":", "."))
    else:
        import wandb
        runwandb = wandb.init(
            project="greedyNeuron2",
            name=name,
            config=dict)


def iter_by_epoch(epoch, step):
    global max_step
    if step is None:
        return (epoch + 1) * max_step

    max_step = max(max_step, step)
    return epoch * max_step + step


def log(group, value, epoch, step=None, silent=False, translate=True):
    global previous_epoch, task_id, forgetting_setting, total_epoches, max_step

    if epoch < previous_epoch:
        task_id += 1
    previous_epoch = epoch
    epoch=epoch+task_id*total_epoches
    if disable:
        return
    if translate:
        iter = iter_by_epoch(epoch, step)
    else:
        iter = epoch
    if writer_mode == "tensorboard":
        writer.add_scalar(group, value, iter)
    else:
        wandb.log({group: value, 'epoch': epoch, 'batch': step})

    if not silent:
        print(f"{group}: {value}, epoch:{epoch} step:{step}")
