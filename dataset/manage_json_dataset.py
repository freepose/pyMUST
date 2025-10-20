#!/usr/bin/env python
# encoding: utf-8

import sys, ujson
import numpy as np

from typing import List, Tuple, Union, Any, Dict, List, Optional, Mapping
from pathlib import Path

from tqdm import tqdm

from must import get_device
from must.data import MultitaskDataset
from must.data_type import SplitRatio

from .json_tools import list_of_dicts_to_dict_of_lists, materialize_input_leaf_tensors

MultiTaskDatasetSequence = Union[MultitaskDataset, Tuple[MultitaskDataset, ...], List[MultitaskDataset]]

json_metadata = {
    'PhysioNetJson': {
        'description': 'PhysioNet Challenge 2012 JSON Dataset. Each line is a JSON clinical record.',
        'fields': {
            'label': 'The target label for the record.',
            'forward': {
                'values': 'The observed values in the forward direction.',
                'masks': 'The masks indicating observed values in the forward direction.',
                'evals': 'The evaluation values in the forward direction.',
                'eval_masks': 'The evaluation masks in the forward direction.',
                'forwards': 'The forward time steps.',
                'deltas': 'The time deltas in the forward direction.'
            },
            'backward': {
                'values': 'The observed values in the backward direction.',
                'masks': 'The masks indicating observed values in the backward direction.',
                'evals': 'The evaluation values in the backward direction.',
                'eval_masks': 'The evaluation masks in the backward direction.',
                'forwards': 'The backward time steps.',
                'deltas': 'The time deltas in the backward direction.'
            }
        },

        'columns': {
            'names': ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose',
                      'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2',
                      'K', 'GCS', 'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets',
                      'Urine', 'NIMAP', 'Creatinine', 'ALP'],
            'targets': ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose',
                        'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2',
                        'K', 'GCS', 'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets',
                        'Urine', 'NIMAP', 'Creatinine', 'ALP'],
        },

        # The files have common data structures
        'files': [
            '{root}/disease/PhysioNet_Challenge_2012/03_json/set-a.json',
            # '{root}/disease/PhysioNet_Challenge_2012/03_json/set-b.json',
            # '{root}/disease/PhysioNet_Challenge_2012/03_json/set-c.json'
        ],

        'classify': {
            'description': 'Configuration for "MultiTaskDataset" loading (forward only)',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}}],
            'outputs': [{'label': 'int'}],
        },

        'forward': {
            'description': 'Configuration for "MultiTaskDataset" loading (forward only)',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}}],
            'outputs': [{'forward': {'evals': 'float32', 'eval_masks': 'bool'}}],
        },

        'classify+forward': {
            'description': 'Configuration for "MultiTaskDataset" loading (forward + backward)',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}}],
            'outputs': [{'label': 'int'}, {'forward': {'evals': 'float32', 'eval_masks': 'bool'}}],
        },

        'classify+forward+forwards': {
            'description': 'Configuration for "MultiTaskDataset" loading (forward + backward)',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32', 'forwards': 'float32'}}],
            'outputs': [{'label': 'int'}, {'forward': {'evals': 'float32', 'eval_masks': 'bool'}}],
        },

        'classify+forward+backward': {
            'description': 'Configuration for "MultiTaskDataset" loading (forward + backward)',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'},
                        'backward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}}],
            'outputs': [{'label': 'int'},
                        {'label': 'int'},
                        {'label': 'int'},
                        {'forward': {'evals': 'float32', 'eval_masks': 'bool'}},
                        # {'backward': {'evals': 'float32', 'eval_masks': 'bool'}}
                        ],
        },

        'config-test': {
            'description': 'the same as config2, but inputs are separated into two parts',
            'inputs': [{'forward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}},
                       {'backward': {'values': 'float32', 'masks': 'bool', 'deltas': 'float32'}}],
            'outputs': [{'label': 'int'},
                        {'forward': {'evals': 'float32', 'eval_masks': 'bool'}},
                        {'backward': {'evals': 'float32', 'eval_masks': 'bool'}}],
        }
    }
}


def prepare_multitask_dataset(data_root: str, dataset_name: str,
                              task_config: str = 'config1',
                              split_ratios: SplitRatio = None,
                              device: str = 'cpu',
                              show_progress: bool = True) -> MultiTaskDatasetSequence:
    """
        Prepare the JSON metadata for MultiTaskDataset.
    """
    assert dataset_name in json_metadata, f"Dataset '{dataset_name}' is not defined in the metadata."
    metadata = json_metadata.get(dataset_name)

    assert 'files' in metadata, f"'files' not defined in the metadata for dataset '{dataset_name}'."
    json_filenames = [f.format(root=data_root) for f in metadata['files']]

    assert task_config in metadata, \
        f"Task config '{task_config}' not defined in the metadata for dataset '{dataset_name}'."
    config = metadata.get(task_config)
    assert 'inputs' in config and 'outputs' in config, \
        f"'inputs' or 'outputs' not defined in the task config '{task_config}'."
    config_inputs, config_outputs = config['inputs'], config['outputs']

    print(f'Loading and parsing', json_filenames)

    records = []
    for filename in json_filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()  # [:500]

        name = Path(filename).name
        with tqdm(total=len(lines), file=sys.stdout, leave=False, disable=not show_progress) as pbar:
            for i, line in enumerate(lines, start=1):
                pbar.set_description(f'\tParsing "{name}" at line {i} / {len(lines)}')
                records.append(ujson.loads(line))
                pbar.update(1)

    print(f'\t(1) Transforming list of dicts to dict of lists ...', end='\t')
    records_dict = list_of_dicts_to_dict_of_lists(records)

    # The inputs are regarded as single-object structures
    # The outputs are regarded as list-of-object structures

    print(f'\t(2) Collecting inputs / outputs via transforming list to torch.Tensor and device ...', end='\t')
    inputs = []
    for i_dict in config_inputs:  # config_inputs is a list of dicts
        materialize_i_dict = materialize_input_leaf_tensors(records_dict, i_dict)
        inputs.extend([v.to(get_device(device)) for _, v in materialize_i_dict.items()])

    outputs = []
    for o_dict in config_outputs:
        materialize_o_dict = materialize_input_leaf_tensors(records_dict, o_dict)
        outputs.append([v.to(get_device(device)) for _, v in materialize_o_dict.items()])

    print(f'\t(3) Loading to "MultiTaskDataset"...')
    dataset = MultitaskDataset(inputs, outputs, shuffle=True, mark=dataset_name)
    if split_ratios is None:
        return dataset

    split_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])
    for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
        split_ds = dataset.split(s, e, mark=f'{dataset_name}_part{i}')
        split_ds.ratio = round(e - s, 15)
        split_datasets.append(split_ds)

    return split_datasets
