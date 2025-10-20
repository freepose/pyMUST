#!/usr/bin/env python
# encoding: utf-8

import torch

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional

from ..data_type import TensorSequence


class AbstractMetric(ABC):
    """
        Abstract class for ** streaming aggregated metrics **.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass


class MultitaskCriteria(AbstractMetric):
    """
        The loss function for multitask learning, which combines multiple criteria.
    """

    def __init__(self, criteria: List[AbstractMetric],
                 weights: Optional[Union[List[float], Tuple[float, ...]]] = None):
        super().__init__()

        assert all(isinstance(c, AbstractMetric) for c in criteria), \
            "All criteria must be instances of 'AbstractMetric'."

        self.criteria = criteria
        self.weights = weights if weights is not None else [1.0] * len(criteria)

        assert len(self.criteria) == len(self.weights), \
            "The number of criteria must match the number of weights."

        self.abbr_names = []
        for i, criterion in enumerate(self.criteria, start=1):
            name = type(criterion).__name__
            if not name.isupper():
                name = ''.join(ch for ch in name if ch.isupper())
            self.abbr_names.append(f'{i}_{name}({round(self.weights[i-1], 10)})')

    def __call__(self, tasks: List[Tuple[torch.Tensor, TensorSequence]]) -> torch.Tensor:
        """
            Compute the combined loss from all criteria for a batch.
            Each task in tasks corresponds to a criterion in self.criteria.
            Each task is a tuple of (y_hat, targets),
            where ``y_hat`` is the model output and targets are the ground truth values.
            The final loss is the weighted sum of individual losses from each criterion.

            :param tasks: A list of tasks, where each task is a tuple of (y_hat, targets).
            :return: The combined loss as a torch.Tensor.
        """
        losses = [weight * criterion(y_hat, *targets) for criterion, weight, (y_hat, targets) in
                  zip(self.criteria, self.weights, tasks)]

        return torch.stack(losses).sum()

    def reset(self):
        """
            Reset all the criteria to its initial state.
        """
        for criterion in self.criteria:
            criterion.reset()

    def update(self, tasks: List[Tuple[torch.Tensor, TensorSequence]]):
        """
            Update all the criteria with a new batch.
        """

        for criterion, (y_hat, targets) in zip(self.criteria, tasks):
            criterion.update(y_hat, *targets)

    def compute(self) -> Dict[str, float]:
        """
            Compute the combined loss from all criteria.
        """
        results = {'loss': 0.}
        for criterion, weight, abbr in zip(self.criteria, self.weights, self.abbr_names):
            results[abbr] = criterion.compute()
            results['loss'] += weight * results[abbr]

        return results
