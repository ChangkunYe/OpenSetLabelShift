import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import step_lr, warmup_step_lr


class MixUp:
    def __init__(self, batch_size, alpha):
        self.weight = np.random.beta(alpha, alpha)
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)

    def augment_input(self, img):
        return self.weight * img + (1 - self.weight) * img[self.new_idx, :]

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat(len(labels) // self.batch_size)
        return self.weight * criterion(logits, labels) + (1 - self.weight) * criterion(logits, new_labels)


class OWLSTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.alpha = self.config.trainer.trainer_args.alpha

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        mile_stones = [100, 150] if config.optimizer.num_epochs >= 200 else [60, 80]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: step_lr(step, mile_stones),
        )
        # cosine_annealing(step, config.optimizer.num_epochs, 1, 1e-6 / config.optimizer.lr)
        # warmup_step_lr(step, mile_stones=mile_stones)

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)

            # mixup operation
            aug_method = MixUp(len(batch['data']), self.alpha)
            data_mix = aug_method.augment_input(batch['data'].cuda())

            # forward
            logits_classifier = self.net(data_mix)
            loss = aug_method.augment_criterion(F.cross_entropy, logits_classifier, batch['label'].cuda())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        self.scheduler.step()
        # print('Current lr:', self.scheduler.get_last_lr())
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics
