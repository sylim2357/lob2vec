from torch.optim import Optimizer
from torch import optim
import torch
from torch.optim.optimizer import Optimizer, required

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g[
                    'weight_decay_filter'
                ] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g[
                    'lars_adaptation_filter'
                ] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g['eta'] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class LARSWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        eta: float = 1e-3,
        clip: bool = False,
        eps: float = 1e-8,
        exclude_bias_n_norm: bool = False,
    ):
        """Wrapper that adds LARS scheduling to any optimizer.
        This helps stability with huge batch sizes.
        Args:
            optimizer (Optimizer): torch optimizer.
            eta (float, optional): trust coefficient. Defaults to 1e-3.
            clip (bool, optional): clip gradient values. Defaults to False.
            eps (float, optional): adaptive_lr stability coefficient. Defaults to 1e-8.
            exclude_bias_n_norm (bool, optional): exclude bias and normalization layers from lars.
                Defaults to False.
        """

        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

        self.__setstate__ = self.optim.__setstate__  # type: ignore
        self.__getstate__ = self.optim.__getstate__  # type: ignore
        self.__repr__ = self.optim.__repr__  # type: ignore

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property  # type: ignore
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, state):
        self.optim.state = state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None and (
                    p.ndim != 1 or not self.exclude_bias_n_norm
                ):
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (
                g_norm + p_norm * weight_decay + self.eps
            )

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr

# class LARS(Optimizer):
#     r"""Implements layer-wise adaptive rate scaling for SGD.
#     Args:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float): base learning rate (\gamma_0)
#         momentum (float, optional): momentum factor (default: 0) ("m")
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#             ("\beta")
#         eta (float, optional): LARS coefficient
#         max_epoch: maximum training epoch to determine polynomial LR decay.
#     Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
#     Large Batch Training of Convolutional Networks:
#         https://arxiv.org/abs/1708.03888
#     Example:
#         >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
#         >>> optimizer.zero_grad()
#         >>> loss_fn(model(input), target).backward()
#         >>> optimizer.step()
#     """
#     def __init__(self, params, lr=required, momentum=.9,
#                  weight_decay=.0001, eta=0.01, max_epoch=200):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}"
#                              .format(weight_decay))
#         if eta < 0.0:
#             raise ValueError("Invalid LARS coefficient value: {}".format(eta))

#         self.epoch = 0
#         defaults = dict(lr=lr, momentum=momentum,
#                         weight_decay=weight_decay,
#                         eta=eta, max_epoch=max_epoch)
#         super(LARS, self).__init__(params, defaults)

#     def step(self, epoch=None, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#             epoch: current epoch to calculate polynomial LR decay schedule.
#                    if None, uses self.epoch and increments it.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         if epoch is None:
#             epoch = self.epoch
#             self.epoch += 1

#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             eta = group['eta']
#             lr = group['lr']
#             max_epoch = group['max_epoch']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 param_state = self.state[p]
#                 d_p = p.grad.data

#                 weight_norm = torch.norm(p.data)
#                 grad_norm = torch.norm(d_p)

#                 # Global LR computed on polynomial decay schedule
#                 decay = (1 - float(epoch) / max_epoch) ** 2
#                 global_lr = lr * decay

#                 # Compute local learning rate for this layer
#                 local_lr = eta * weight_norm / \
#                     (grad_norm + weight_decay * weight_norm)

#                 # Update the momentum term
#                 actual_lr = local_lr * global_lr

#                 if 'momentum_buffer' not in param_state:
#                     buf = param_state['momentum_buffer'] = \
#                             torch.zeros_like(p.data)
#                 else:
#                     buf = param_state['momentum_buffer']
#                 buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
#                 p.data.add_(-buf)

#         return loss