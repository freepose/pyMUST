#!/usr/bin/env python
# encoding: utf-8

import torch

from typing import Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod

from setuptools.namespaces import flatten

from ..data_type import TensorSequence

from .regress import MSE, MAE, MRE, RMSE, MAPE, CVRMSE, SMAPE, PCC, RAE, RSE, R2
from .classify import CrossEntropy, Accuracy, Precision, Recall, AUC


class AbstractEvaluator(ABC):
    """
        Abstract class for streaming aggregated evaluating forecasting models on large-scale datasets.
    """

    def __init__(self, *args, **kwargs):
        # self.metrics = {}  # format: {metric_name: metric_class, ...}, default is empty
        pass

    @abstractmethod
    def reset(self):
        """ Reset the metrics to its initial state. """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Update the metrics with a new batch of <prediction tensor, real tensor, or mask tensor> pair. """
        pass

    @abstractmethod
    def compute(self) -> Dict:
        """ Compute the evaluation metrics. """
        pass


class EmptyEvaluator(AbstractEvaluator):
    """
        No-operation evaluator for debugging or disabling evaluation.

        This evaluator implements the AbstractEvaluator interface but performs no actual evaluation operations.
        Useful for:
            - Debugging scenarios where evaluation should be skipped
            - Performance testing without evaluation overhead
            - Placeholder during development

        All methods are no-ops and return empty results.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self) -> Dict:
        return {}


class RegressEvaluator(AbstractEvaluator):
    """
        The ``Evaluator()`` class manages the metrics for both the loss function and evaluation metrics.

        The metrics support both complete and incomplete time series.

        :param metrics: List of **metric names** to use. If None, use all available metrics.
        :param metric_params: Dictionary of metric names and their additional parameters, e.g., PCC bias.
                              {'PCC': {'bias': 0.001}}.
    """

    def __init__(self, metrics: Union[List[str], Tuple[str, ...]] = None,
                 metric_params: Optional[Dict[str, Dict]] = None):
        super().__init__()

        self.available_metrics = {
            'MSE': MSE,
            'MAE': MAE,
            'MRE': MRE,
            'RMSE': RMSE,
            'MAPE': MAPE,
            'sMAPE': SMAPE,
            'CV-RMSE': CVRMSE,
            'PCC': PCC,
            'RAE': RAE,
            'RSE': RSE,
            'R2': R2,
            # Add more metrics here as needed
        }

        if metrics is None:
            self.metrics = self.available_metrics
        else:
            # Ensure all specified metrics are valid
            assert all(metric in self.available_metrics for metric in metrics), \
                f'Metrics should be in {list(self.available_metrics.keys())}'

            # Filter the available metrics based on the provided list
            self.metrics = {metric: self.available_metrics[metric] for metric in metrics if
                            metric in self.available_metrics}

        self.metric_params = metric_params if metric_params else {}

        self.metric_instances = {name: metric_class(**self.metric_params.get(name, {}))
                                 for name, metric_class in self.metrics.items()}

    def reset(self):
        """ Reset all metrics to their initial state. """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.reset()

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metrics with a new batch of <prediction tensor, real tensor, or mask tensor> pair.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.update(prediction, real, mask)

    def compute(self) -> Dict:
        """
            Evaluate the prediction performance of error metrics.
            :return: Dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_instances.items():
            ret = metric_inst.compute()
            results[name] = float(ret)
        return results

    def evaluate(self, *tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> Dict:
        """
            Evaluate the prediction performance using the error / accuracy metrics.

            :param tensors: prediction tensor, real value tensor, (maybe) mask tensor (1d, 2d, or 3d tensor)
            :return: dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_instances.items():
            params = self.metric_params.get(name, {})
            ret = metric_inst(*tensors, **params)
            results[name] = float(ret)
        return results


class ClassifyEvaluator(AbstractEvaluator):
    """
        The ``ClassifyEvaluator()`` class manages the metrics for classification tasks.

        :param metrics: List of **metric names** to use. If None, use all available metrics.
        :param metric_params: Dictionary of metric names and their additional parameters.
    """

    def __init__(self, metrics: Union[List[str], Tuple[str, ...]] = None,
                 metric_params: Optional[Dict[str, Dict]] = None):
        super().__init__()

        self.available_metrics = {
            'CE': CrossEntropy,
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'AUC': AUC,
            # 'F1Score': F1Score,
        }

        if metrics is None:
            self.metrics = self.available_metrics
        else:
            # Ensure all specified metrics are valid
            assert all(metric in self.available_metrics for metric in metrics), \
                f'Metrics should be in {list(self.available_metrics.keys())}'

            # Filter the available metrics based on the provided list
            self.metrics = {metric: self.available_metrics[metric] for metric in metrics if
                            metric in self.available_metrics}

        self.metric_params = metric_params if metric_params else {}
        self.metric_instances = {name: metric_class(**self.metric_params.get(name, {}))
                                 for name, metric_class in self.metrics.items()}

    def reset(self):
        """ Reset all metrics to their initial state. """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.reset()

    def update(self, prediction: torch.Tensor, real: torch.Tensor):
        """
            Update the metrics with a new batch of <prediction tensor, real tensor, or mask tensor> pair.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
        """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.update(prediction, real)

    def compute(self) -> Dict:
        """
            Evaluate the prediction performance of error metrics.
            :return: Dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_instances.items():
            ret = metric_inst.compute()
            results[name] = float(ret)

        return results


class MultitaskEvaluator(AbstractEvaluator):
    """
        The ``MultiTaskEvaluator()`` class manages multiple evaluators for multitask learning scenarios.

        :param evaluators: Dictionary of task names and their corresponding evaluator instances.
    """

    def __init__(self, evaluators: Dict[str, AbstractEvaluator]):
        super().__init__()

        assert all(isinstance(evaluator, AbstractEvaluator) for evaluator in evaluators.values()), \
            "All evaluators must be instances of 'AbstractEvaluator'."

        self.evaluators = evaluators
        self.metric_names = []
        for task_name, evaluator in self.evaluators.items():
            for metric_name in evaluator.compute().keys():
                # For metric_name, if is upper case, keep it as is; else, use the first letter uppercase,
                # if the second letter is digit, keep it as first two letters uppercase
                if metric_name.isupper():
                    abbr_metric_name = metric_name
                elif len(metric_name) > 1 and metric_name[1].isdigit():
                    abbr_metric_name = metric_name[:2].upper()
                else:
                    abbr_metric_name = metric_name[0].upper()
                self.metric_names.append(f"{task_name}_{abbr_metric_name}")

    def reset(self):
        """ Reset all evaluators to their initial state. """
        for evaluator in self.evaluators.values():
            evaluator.reset()

    def update(self, tasks: List[Tuple[torch.Tensor, TensorSequence]]):
        """
            Update each evaluator with a new batch of <prediction tensor, real tensor, or mask tensor> pair
            for each supervised task.

            :param tasks: List of tuples, each containing (prediction tensor, real tensor, (maybe) mask tensor)
                                     for each task.
        """
        for (task_name, evaluator), (y_hat, targets) in zip(self.evaluators.items(), tasks):
            evaluator.update(y_hat, *targets)

    def compute(self) -> Dict[str, Dict]:
        """
            Evaluate the prediction performance of all tasks.
            :return: Dictionary of task names and their corresponding metric results.
        """
        flattened_results = {}
        for task_name, evaluator in self.evaluators.items():
            task_results = evaluator.compute()  # ``EmptyEvaluator`` returns {}
            for metric_name, value in task_results.items():
                flattened_results[f"{task_name}_{metric_name}"] = value

        return flattened_results
