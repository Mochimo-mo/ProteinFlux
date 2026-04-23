# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/exponential_moving_average.py
from collections import OrderedDict
import torch
import torch.nn as nn

from core.tensor_utils import tensor_tree_map


class ExponentialMovingAverage:
    """
    Maintains moving averages of parameters with exponential decay.

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`
    """

    def __init__(self, model: nn.Module, decay: float):
        super(ExponentialMovingAverage, self).__init__()
        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

    def to(self, device):
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update, state_dict):
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        for k in state_dict["params"].keys():
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict({"params": self.params, "decay": self.decay})
