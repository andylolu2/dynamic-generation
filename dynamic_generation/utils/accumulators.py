import abc

import torch


class Accumulator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, init_value):
        ...

    @abc.abstractmethod
    def update(self, value):
        ...

    @abc.abstractmethod
    def compute(self):
        ...


class ReplaceAccumulator(Accumulator):
    def __init__(self, init_value):
        self.value = init_value

    def update(self, value):
        self.value = value

    def compute(self):
        return self.value


class MeanAccumulator(Accumulator):
    def __init__(self, init_value, ignore_nan=True):
        self.value = torch.as_tensor(init_value, dtype=torch.float32)
        self.count = torch.tensor(1.0)
        self.ignore_nan = ignore_nan

    def update(self, value):
        value = torch.as_tensor(value, dtype=self.value.dtype)
        if self.ignore_nan:
            value = value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        self.value = self.value + value
        self.count = self.count + 1

    def compute(self):
        res = self.value / self.count
        if res.numel() == 1:
            res = res.squeeze()
        return res.cpu().numpy()


class SumAccumulator(Accumulator):
    def __init__(self, init_value, ignore_nan=True):
        self.value = torch.as_tensor(init_value)
        self.ignore_nan = ignore_nan

    def update(self, value):
        value = torch.as_tensor(value, dtype=self.value.dtype)
        if self.ignore_nan:
            value = value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        self.value = self.value + value

    def compute(self):
        return self.value.cpu().numpy()


class MinAccumulator(Accumulator):
    def __init__(self, init_value):
        self.value = torch.as_tensor(init_value)

    def update(self, value):
        self.value = torch.minimum(self.value, value)

    def compute(self):
        return self.value.cpu().numpy()


class MaxAccumulator(Accumulator):
    def __init__(self, init_value):
        self.value = torch.as_tensor(init_value)

    def update(self, value):
        self.value = torch.maximum(self.value, value)

    def compute(self):
        return self.value.cpu().numpy()


class StackAccumulator(Accumulator):
    def __init__(self, init_value=None, batched=False):
        self.value = []
        self.batched = batched

        if init_value is not None:
            self.update(init_value)

    def update(self, value):
        self.value.append(torch.as_tensor(value))

    def compute(self):
        if self.batched:
            res = torch.concat(self.value)
        else:
            res = torch.stack(self.value)
        return res.cpu().numpy()
