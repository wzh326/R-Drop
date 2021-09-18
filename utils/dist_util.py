import paddle.distributed as dist


def is_main_process():
    return dist.get_rank() == 0

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s