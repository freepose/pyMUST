#!/usr/bin/env python
# encoding: utf-8

import os

import torch

from must import initial_seed, initial_logger, get_device
from must.train import MultitaskTrainer
from must.metric import MultitaskEvaluator, EmptyEvaluator, MultitaskCriteria, RegressEvaluator, ClassifyEvaluator
from must.metric import CrossEntropy, MSE, MAE, MRE, ORAE

from must.model.base.utils import get_model_info
from must.model.rits import RITS, RITSI
from must.model.m_rnn import MRNN
from must.model.gru_d import GRUD
from must.model.brits import BRITSI, BRITS

from dataset.manage_json_dataset import prepare_multitask_dataset


def function_based_methods():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    ds_device, model_device = 'cpu', 'mps'

    train_ds, val_ds, test_ds = prepare_multitask_dataset(data_root, 'PhysioNetJson', 'function-based', (0.6, 0.2, 0.2), ds_device)

    # Global-mean method
    train_values, train_masks = train_ds.inputs
    train_mean = (train_values[train_masks]).mean()

    val_evals, val_eval_masks = val_ds.outputs[0]
    test_evals, test_eval_masks = test_ds.outputs[0]

    val_hat = torch.ones_like(val_evals) * train_mean
    val_mae = MAE()(val_hat, val_evals, val_eval_masks)
    val_orae = ORAE()(val_hat, val_evals, val_eval_masks)

    test_hat = torch.ones_like(test_evals) * train_mean
    test_mae = MAE()(test_hat, test_evals, test_eval_masks)
    test_orae = ORAE()(test_hat, test_evals, test_eval_masks)

    print('global means', {'val_mae': val_mae.item(), 'val_orae': val_orae.item()})
    print('global means', {'test_mae': test_mae.item(), 'test_orae': test_orae.item()})

    # Variable-mean method
    sum_vals = (train_values * train_masks).sum(dim=[0, 1])
    count_vals = train_masks.sum(dim=[0, 1])
    feature_means = sum_vals / count_vals.clamp(min=1)

    val_hat = feature_means.expand_as(val_evals)
    val_mae = MAE()(val_hat, val_evals, val_eval_masks)
    val_orae = ORAE()(val_hat, val_evals, val_eval_masks)

    test_hat = feature_means.expand_as(test_evals)
    test_mae = MAE()(test_hat, test_evals, test_eval_masks)
    test_orae = ORAE()(test_hat, test_evals, test_eval_masks)

    print('feature means',{'val_mae': val_mae.item(), 'val_orae': val_orae.item()})
    print('feature means',{'test_mae': test_mae.item(), 'test_orae': test_orae.item()})

    # time-mean method
    sum_vals = (train_values * train_masks).sum(dim=[0, 2])
    count_vals = train_masks.sum(dim=[0, 2])
    time_means = sum_vals / count_vals.clamp(min=1)

    val_hat = time_means.expand_as(val_evals.permute(0, 2, 1)).permute(0, 2, 1)
    val_mae = MAE()(val_hat, val_evals, val_eval_masks)
    val_orae = ORAE()(val_hat, val_evals, val_eval_masks)

    test_hat = time_means.expand_as(test_evals.permute(0, 2, 1)).permute(0, 2, 1)
    test_mae = MAE()(test_hat, test_evals, test_eval_masks)
    test_orae = ORAE()(test_hat, test_evals, test_eval_masks)

    print('time means', {'val_mae': val_mae.item(), 'val_orae': val_orae.item()})
    print('time means', {'test_mae': test_mae.item(), 'test_orae': test_orae.item()})



def main():
    """"
        For 'classify+impute' multitask, you should prepare:

        (1) Dataset: with both classification labels and imputation evals.
        (2) Model: that can output both classification logits and imputed values.
        (3) Criteria: with both classification loss and regression loss.
        (4) Evaluator: with both classification metrics and regression metrics.

    """
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    ds_device, model_device = 'cpu', 'mps'

    # classify+forward+forwards
    # train_ds, val_ds, test_ds = prepare_multitask_dataset(data_root, 'PhysioNetJson', 'classify+forward+backward', (0.6, 0.2, 0.2), ds_device, show_progress=True)
    # train_ds, val_ds, test_ds = prepare_multitask_dataset(data_root, 'PhysioNetJson', 'classify+forward+forwards', (0.6, 0.2, 0.2), ds_device, show_progress=True)
    train_ds, val_ds, test_ds = prepare_multitask_dataset(data_root, 'PhysioNetJson', 'classify+forward+backward',
                                                          (0.6, 0.2, 0.2), ds_device, show_progress=True)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    # inputs: forward only, with values, masks, deltas
    # model = RITS(input_size=35, rnn_hidden_size=64, n_classes=2, recovery_weight=0.1)
    # model = RITSI(input_size=35, rnn_hidden_size=64, n_classes=2, recovery_weight=0.1)

    # inputs: forward + backward, each with values, masks, deltas
    # model = MRNN(input_size=35, rnn_hidden_size=64, n_classes=2, recovery_weight=0.1)

    # inputs: forward only, with values, masks, deltas, forwards
    # model = GRUD(input_size=35, rnn_hidden_size=64, n_classes=2, dropout_rate=0.5)

    # inputs: forward + backward, each with values, masks, deltas
    model = BRITSI(input_size=35, rnn_hidden_size=64, n_classes=2, recovery_weight=0.5, consistency_weight=0.1)
    # model = BRITS(input_size=35, rnn_hidden_size=108, n_classes=2, dropout_rate=0.25, recovery_weight=0.3)

    print(get_model_info(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=0.001)
    criteria = MultitaskCriteria([CrossEntropy(), CrossEntropy(), CrossEntropy(), MSE()], weights=[1.0, 1.0, 0., 0.])
    evaluator = MultitaskEvaluator({'f': EmptyEvaluator(),
                                    'b': EmptyEvaluator(),
                                    't': ClassifyEvaluator(['AUC', 'Accuracy']),
                                    'im': RegressEvaluator(['MAE', 'ORAE'])})

    trainer = MultitaskTrainer(get_device(model_device), model,
                               is_initial_weights=True, is_compile=False,
                               optimizer=optimizer,
                               criteria=criteria, evaluator=evaluator)

    trainer.fit(train_ds, val_ds, 32,
                epoch_range=(1, 1000), shuffle=True)

    if test_ds is not None:
        test_metrics = trainer.evaluate(test_ds, 32)
        print(f'Test metrics: {test_metrics}')
    elif val_ds is not None:
        val_metrics = trainer.evaluate(val_ds, 32)
        print(f'Validation metrics: {val_metrics}')


if __name__ == '__main__':
    initial_seed(42)
    initial_logger()

    main()
    # function_based_methods()
