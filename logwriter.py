from torch.utils.tensorboard import SummaryWriter
import datetime

# Writer will output to ./runs/ directory by default
writer = None
max_step = 0
writer_mode = None
disable = False


def start_writer(name, mode="tensorboard", dict=None):
    if disable:
        return
    global writer, writer_mode
    writer_mode = mode
    if mode == "tensorboard":
        writer = SummaryWriter("runs/" + name + "_" + str(datetime.datetime.now()).replace(":", "."))
    else:
        import wandb
        runwandb = wandb.init(dict)


def iter_by_epoch(epoch, step):
    global max_step
    if step is None:
        return (epoch + 1) * max_step

    max_step = max(max_step, step)
    return epoch * max_step + step


def log(group, value, epoch, step=None, silent=False, translate=True):
    if disable:
        return
    if translate:
        iter = iter_by_epoch(epoch, step)
    else:
        iter = epoch
    if writer_mode == "tensorboard":
        writer.add_scalar(group, value, iter)
    else:
        raise NotImplementedError
    if not silent:
        print(f"{group}: {value}, epoch:{epoch} step:{step}")
