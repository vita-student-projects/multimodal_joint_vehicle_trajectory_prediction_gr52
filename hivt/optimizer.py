from typing import List

import torch
from torch import optim

from trajpred.config import Config

from torch.optim.lr_scheduler import MultiStepLR


class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, config: Config, coef=None):

        self.config = config
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            # param_groups.append({"params": param, "lr": 0})
            param_groups.append({"params": param, "lr": config.hydra.model.train.learning_rate})

        opt = config.hydra.model.train.optimizer.lower()
        assert opt == "adam" # it only supports adam. you can add a new optimizer by adding to the config file
        # self.opt = optim.Adam(param_groups, weight_decay=0)
        # self.opt = optim.Adam(param_groups, lr=config.hydra.train.learning_rate, eps=0.0001)
        self.opt = optim.Adam(param_groups, eps=0.0001)

        # self.lr_func = StepLR(config.hydra.train.lrs, config.hydra.train.lr_step_iters) #config.lr_func
        if config.hydra.model.train.scheduler == "multistep":
            self.lr_func = MultiStepLR(self.opt, milestones=config.hydra.model.train.learning_rate_sched, gamma=0.5,
                                               verbose=True)
        elif config.hydra.model.train.scheduler == "plateau":
            self.lr_func = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=5)
        else:
            raise NotImplementedError

        if hasattr(config, "clip_grads"):
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        if self.clip_grads:
            self.clip()
        self.opt.step()

        # From Autobot
        lr = self.opt.param_groups[0]['lr']

        return lr
        # return self.opt.step()

    def scheduler_step(self, val_metric=0):
        if self.config.hydra.model.train.scheduler == "multistep":
            return self.lr_func.step()
        else:
            return self.lr_func.step(val_metric)


    def state_dict(self) -> dict:
        return self.opt.state_dict()

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lrs: List[float], step_iters: List[int]) -> None:
        assert (len(lrs) == len(step_iters) + 1)
        self._lrs = lrs
        self._step_iters = step_iters
        self._iter = 0

    def __call__(self):
        self._iter += 1
        for i, step_iter in enumerate(self._step_iters):
            if self._iter % 1000 == 0:
                print("INFO: _iter", self._iter, step_iter)
            if self._iter <= step_iter:
                return self._lrs[i]
        return self._lrs[-1]
