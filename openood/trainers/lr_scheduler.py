import numpy as np


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))


def step_lr(step, mile_stones=None, gamma=0.1):
    if mile_stones is None:
        mile_stones = [100, 150]
    lr = np.power(gamma, np.digitize([step], mile_stones)[0])
    # print('Scheduler step: ', step, ', lr: ', lr)
    return lr


def warmup_step_lr(step, mile_stones=None, warmup_epoch=5, gamma=0.1):
    if mile_stones is None:
        mile_stones = [100, 150]
    lr = np.power(gamma, np.digitize([step], mile_stones)[0])

    """Warmup"""
    if step < warmup_epoch:
        lr = float(1 + step) / warmup_epoch
    return lr

