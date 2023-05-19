import datetime

writer = None
max_step = 1
writer_mode = None
previous_epoch = -1
task_id = 0
forgetting_setting = False
total_epoches = None

def finsih():
    print("program ends")

def start_writer(forgetting=False, epoches=None, steps=None):
    if not forgetting:
        steps=1
    global writer, writer_mode, forgetting_setting, total_epoches, max_step
    total_epoches = epoches
    max_step= steps
    forgetting_setting = forgetting



def iter_by_epoch(epoch, step):
    global max_step
    if step is None:
        return (epoch + 1) * max_step

    max_step = max(max_step, step)
    return epoch * max_step + step


def log(group, value, epoch, step=None, silent=False, translate=True):
    global previous_epoch, task_id, forgetting_setting, total_epoches, max_step

    if forgetting_setting:
        if epoch < previous_epoch:
            task_id += 1
        previous_epoch = epoch
        epoch=((epoch+task_id*total_epoches)/total_epoches)*200

    print(f"{group}: {value}, epoch:{epoch} step:{step}")
