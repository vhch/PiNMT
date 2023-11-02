# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Collection
from dataclasses import dataclass, field
from typing import List

from fairseq.dataclass import FairseqDataclass
from omegaconf import II, DictConfig

from . import FairseqLRScheduler, register_lr_scheduler


@dataclass
class InverseSquareRootScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is args.lr"
        },
    )
    # TODO common vars at parent class
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("inverse_sqrt", dataclass=InverseSquareRootScheduleConfig)
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, cfg: DictConfig, optimizer):
        super().__init__(cfg, optimizer)

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        if isinstance(cfg.lr, Collection) and len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with inverse_sqrt."
                " Consider --lr-scheduler=fixed instead."
            )
        warmup_end_lr = (
            cfg.lr[0]
            if isinstance(cfg.lr, Collection)
            else cfg.lr
        )
        if cfg.warmup_init_lr < 0:
            cfg.warmup_init_lr = (
                0 if cfg.warmup_updates > 0 else warmup_end_lr
            )
            self.init_lrs = [float(0) / float(max(1, cfg.warmup_updates)) * lr for lr in self.base_lrs]

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (
            warmup_end_lr - cfg.warmup_init_lr
        ) / cfg.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * cfg.warmup_updates ** 0.5


        # initial learning rate
        self.lr = cfg.warmup_init_lr
        # self.optimizer.set_lr(self.lr)
        self.optimizer.set_lr(self.init_lrs)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    # def step_update(self, num_updates):
    #     """Update the learning rate after each update."""
    #     if num_updates < self.cfg.warmup_updates:
    #         self.lr = self.cfg.warmup_init_lr + num_updates * self.lr_step
    #     else:
    #         self.lr = self.decay_factor * num_updates ** -0.5
    #     self.optimizer.set_lr(self.lr)
    #     return self.lr

    def rate_scheduled(self, current_step: int, T1=2000, T2=4000):
        if current_step < T1:
            return float(current_step) / float(max(1, T1))
        elif T1 <= current_step < T2:
            return float(T2 - current_step) / float(max(1, T2 - T1))
        else:
            return 0

    # def step_update(self, num_updates):
    #     """Update the learning rate after each update."""
    #     self.lrs = []
    #     for n, lr in enumerate(self.base_lrs):
    #         if n < 257:
    #             if num_updates < self.cfg.warmup_updates:
    #                 self.lrs.append(float(num_updates) / float(max(1, self.cfg.warmup_updates)) * lr)
    #             else:
    #                 self.lrs.append((self.cfg.warmup_updates ** 0.5) * (num_updates ** -0.5) * lr)
    #         else:
    #             if num_updates < self.cfg.warmup_updates:
    #                 self.lrs.append(float(num_updates) / float(max(1, self.cfg.warmup_updates)) * lr * self.rate_scheduled(current_step=num_updates))
    #             else:
    #                 self.lrs.append((self.cfg.warmup_updates ** 0.5) * (num_updates ** -0.5) * lr * self.rate_scheduled(current_step=num_updates))
    #
    #     self.optimizer.set_lr(self.lrs)
    #     return self.lrs[0]

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.warmup_updates:
            self.lrs = [float(num_updates) / float(max(1, self.cfg.warmup_updates)) * lr for lr in self.base_lrs]
        else:
            self.lrs = [(self.cfg.warmup_updates ** 0.5) * (num_updates ** -0.5) * lr for lr in self.base_lrs]

        self.optimizer.set_lr(self.lrs)
        return self.lrs[0]
