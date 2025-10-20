#!/usr/bin/env python
# encoding: utf-8

import torch
from typing import Union, Tuple, List

SplitRatio = Union[int, float, Tuple[float, ...], List[float]]

SoftTensorSequence = Union[List[torch.Tensor]]
TensorSequence = Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
TensorOrSequence = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
