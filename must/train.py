#!/usr/bin/env python
# encoding: utf-8

import sys, platform, os

import torch
import torch.nn as nn
import torch.utils.data as data

from typing import Tuple, List, Literal, Optional, Dict
from tqdm import tqdm

from .model.base.utils import init_weights, to_string
from .data.mt_dataset import MultitaskDataset
from .metric import MultitaskCriteria, MultitaskEvaluator, ClassifyEvaluator, RegressEvaluator, EmptyEvaluator
from .metric import CrossEntropy, MSE


class MultitaskTrainer:
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 is_initial_weights: bool = False,
                 is_compile: bool = False,
                 optimizer=None, lr_scheduler=None,
                 criteria: MultitaskCriteria = None,
                 evaluator: MultitaskEvaluator = None):

        self.device = device
        self.model = model

        self.is_initial_weights = is_initial_weights
        self.is_compile = is_compile  # MPS device may not support for this compiling.

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criteria = criteria
        self.evaluator = evaluator

        if self.is_initial_weights:
            model.apply(init_weights)

        self.initialize_device()

        if self.is_compile:
            # MPS device may not support for this compiling.
            self.model = torch.compile(self.model)

        if self.optimizer is None:
            model_params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(model_params, lr=0.0001)

        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def initialize_device(self):
        """ Initialize model accelerator. """

        if self.device is not None:  # reset all the model parameters to the device
            self.model = self.model.to(self.device)

        if self.device.type == 'cpu':
            if platform.machine() in ('x86_64', 'AMD64'):
                torch.set_num_threads(os.cpu_count() - 2)
            elif platform.machine() == 'arm64':
                torch.set_num_threads(os.cpu_count())
        elif self.device.type == 'cuda':
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            visible_gpus = [int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip()]
            if len(visible_gpus) > 1 and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=visible_gpus)
        elif self.device.type == 'mps':
            pass

    def run_epoch(self, dataloader: data.DataLoader[MultitaskDataset],
                  mode: Literal['train', 'online', 'val'] = 'train',
                  progress_status: str = None):

        self.model.train() if mode in ['train', 'online'] else self.model.eval()

        self.criteria.reset()
        self.evaluator.reset()

        with tqdm(total=len(dataloader), leave=False, file=sys.stdout, disable=progress_status is None) as pbar:
            pbar.set_description(progress_status)

            for i, (batch_x, batch_y_list) in enumerate(dataloader):
                pbar.set_description(progress_status)

                if self.device != dataloader.dataset.device:
                    batch_x = [x.to(self.device) for x in batch_x]
                    batch_y_list = [[y.to(self.device) for y in task_y_list] for task_y_list in batch_y_list]

                model_outputs = self.model(*batch_x)  # -> predictions of several tasks
                if isinstance(model_outputs, torch.Tensor):
                    model_outputs = [model_outputs]

                supervised_task_list = list(zip(model_outputs, batch_y_list))
                batch_loss = self.criteria(supervised_task_list)
                if getattr(self.model, 'input_aware_loss', None) is not None:
                    batch_loss += self.model.input_aware_loss

                self.criteria.update(supervised_task_list)
                self.evaluator.update(supervised_task_list)

                if mode in ['train', 'online']:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                pbar.set_postfix(batch_loss=batch_loss.item())
                pbar.update(1)

        epoch_losses = self.criteria.compute()
        epoch_results = self.evaluator.compute()

        return {**epoch_losses, **epoch_results}

    def message_header(self, has_val_dataset: bool = False):
        """
            :return: message header.
        """

        metric_names = ['loss'] + self.criteria.abbr_names + self.evaluator.metric_names
        header = ['train_{}'.format(k) for k in metric_names]
        if has_val_dataset:
            header += ['val_{}'.format(k) for k in metric_names]
        header = ['lr'] + header

        return 'epoch\t' + '\t'.join(['{:^10}'.format(s) for s in header])

    def fit(self, train_dataset: MultitaskDataset,
            val_dataset: Optional[MultitaskDataset] = None,
            batch_size: int = 32, shuffle: bool = True,
            epoch_range: Tuple[int, int] = (1, 20),
            show_progress: bool = True):

        train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle)
        val_dataloader = None if val_dataset is None else data.DataLoader(val_dataset, batch_size, False)

        message_header = self.message_header(val_dataset is not None)
        print(message_header)
        for epoch in range(epoch_range[0], epoch_range[1] + 1):
            epoch_status = '{}/{}'.format(epoch, epoch_range[1])
            progress_status = ('training ' + epoch_status) if show_progress else None
            message = [epoch_status, self.optimizer.param_groups[0]['lr']]

            train_losses: dict = self.run_epoch(train_dataloader, 'train', progress_status)
            message += list(train_losses.values())

            if val_dataset is not None:
                val_losses: dict = self.run_epoch(val_dataloader, 'val', 'validating ' + epoch_status)
                message += list(val_losses.values())

            print(to_string(*message))

    def evaluate(self, dataset: MultitaskDataset, batch_size: int = 32,
                 show_progress: bool = True) -> Dict[str, float]:
        """
            Evaluate the model on the given dataset.
            :param dataset: the dataset.
            :param batch_size: the batch size of dataloader.
            :param show_progress: whether to show the progress bar.
            :return: a dictionary of evaluation metrics.
        """
        test_dataloader = data.DataLoader(dataset, batch_size, False)
        progress_status = 'evaluating' if show_progress else None
        test_losses: dict = self.run_epoch(test_dataloader, 'val', progress_status)
        return test_losses
