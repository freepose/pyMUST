#!/usr/bin/env python
# encoding: utf-8

import random

import torch
import torch.utils.data as data

from typing import List, Optional, Union

from ..data_type import TensorOrSequence


class MultitaskDataset(data.Dataset):
    def __init__(self, inputs: TensorOrSequence, outputs: List[TensorOrSequence],
                 shuffle: bool = False,
                 mark: Optional[str] = None):
        """
            Multi-Task Dataset for time series imputation tasks.

            :param inputs: List of input tensors.
            :param outputs: List of list output tensors. Each sublist corresponds to a task.
            :param shuffle: Whether to shuffle the dataset for ``split()``.
            :param mark: An optional mark for the dataset.
        """
        super(MultitaskDataset, self).__init__()

        # Ensure inputs and outputs are in list format, and outputs is a list of lists
        self.inputs = [inputs] if isinstance(inputs, torch.Tensor) else inputs
        self.outputs = [[o] if isinstance(o, torch.Tensor) else o for o in outputs]

        self.mark = mark
        self.ratio = 1.

        self.samples = self.inputs[0].shape[0]
        # self.tasks_inputs = len(self.inputs)
        # self.task_outputs = [1 if isinstance(o, torch.Tensor) else len(o) for o in self.outputs]
        self.device = self.inputs[0].device

        self.in_dims = [inp.shape[-1] if inp.ndim > 1 else 1 for inp in self.inputs]
        self.out_dims = [[out.shape[-1] if out.ndim > 1 else 1 for out in task_out] for task_out in self.outputs]

        if shuffle:
            indices = torch.randperm(self.samples, dtype=torch.long)
            self.inputs = [inp[indices] for inp in self.inputs]
            self.outputs = [[out[i][indices] for i in range(len(out))] for out in self.outputs]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        """
            Get a sample by index.

            :param index: The index of the sample to retrieve.
            :return: A tuple containing the input tensors and output tensors for the specified index.
        """
        input_sample = [inp[index] for inp in self.inputs]
        output_sample = [[out[index] for out in task_out] for task_out in self.outputs]

        return input_sample, output_sample

    def __str__(self):
        """
            String representation of the dataset.
        """
        params = dict()
        params['device'] = str(self.device)
        params['ratio'] = self.ratio
        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'ratio': self.ratio,
            'mark': self.mark,
            'samples': len(self),
            'in_dims': self.in_dims,
            'out_dims': self.out_dims
        })

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'MultitaskDataset({})'.format(params_str)

        return params_str

    def split(self, start_ratio: float, end_ratio: float, mark: Optional[str] = None) -> 'MultitaskDataset':
        """
            Split the dataset into a new dataset based on the specified start and end ratios.

            :param start_ratio: The starting ratio for the split (inclusive).
            :param end_ratio: The ending ratio for the split (exclusive).
            :param mark: An optional mark for the dataset.

            :return: A new MultiTaskDataset instance representing the split dataset.
        """
        assert 0.0 <= start_ratio < end_ratio <= 1.0, \
            f"Invalid split ratios: start_ratio={start_ratio}, end_ratio={end_ratio}."

        start_index = int(round(self.samples * start_ratio, 15))
        end_index = int(round(self.samples * end_ratio, 15))

        split_inputs = [inp[start_index:end_index] for inp in self.inputs]
        split_outputs = [[out[start_index:end_index] for out in task_out] for task_out in self.outputs]
        split_dataset = MultitaskDataset(split_inputs, split_outputs, shuffle=False, mark=mark)
        split_dataset.ratio = end_ratio - start_ratio

        return split_dataset