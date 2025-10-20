
import torch

from typing import List, Tuple, Union, Any, Dict, List, Optional, Mapping

TORCH_DTYPE_MAP = {
    'float64': torch.float64,
    'float32': torch.float32,
    'int': torch.long,
    'bool': torch.bool,
}

def list_of_dicts_to_dict_of_lists(records: List[Dict[str, Any]], fill_missing: Optional[Any] = None) -> Dict[str, Any]:
    """
        Convert a list of dictionaries into a dictionary of lists, recursively.

        Rules:
        - Align all keys across records; missing keys are filled with `fill_missing`.
        - If a column is entirely dicts (ignoring None), recurse on that column.
        - Non-dict values (scalars, lists, numbers, strings, etc.) are collected as-is.

        Example:
        [
          {"a": 1, "b": {"x": 10}},
          {"a": 2, "b": {"x": 20}}
        ]
        ->
        {"a": [1, 2], "b": {"x": [10, 20]}}

        Args:
            records: List of dictionaries to convert.
            fill_missing: Placeholder for missing keys; default is None.

        Returns:
            A dictionary of lists, with nested dictionaries converted recursively.
    """
    if not records:
        return {}

    for idx, d in enumerate(records):
        if not isinstance(d, dict):
            raise TypeError(f"Each element must be a dict, but got {type(d)} at index {idx}")

    # Collect the union of all keys to align columns
    all_keys = set()
    for d in records:
        all_keys.update(d.keys())

    out: Dict[str, Any] = {}
    for k in all_keys:
        col = [d.get(k, fill_missing) for d in records]

        # Decide whether to recurse: all non-None values must be dicts
        non_none = [v for v in col if v is not None]
        if non_none and all(isinstance(v, dict) for v in non_none):
            # Treat None as empty dict to preserve alignment during recursion
            nested_records = [v if isinstance(v, dict) else {} for v in col]
            out[k] = list_of_dicts_to_dict_of_lists(nested_records, fill_missing=fill_missing)
        else:
            out[k] = col

    return out


def _get_nested(record: Mapping[str, Any], path: List[str]) -> Any:
    cur: Any = record
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            joined = "/".join(path)
            raise KeyError(f"Missing key along path: {joined}")
        cur = cur[k]
    return cur


def _fill_none(x: Any, leaf_type: type) -> Any:
    # 递归将 None 替换为与目标 dtype 兼容的占位
    if x is None:
        if leaf_type is float:
            return float("nan")
        if leaf_type is int:
            return -1
        if leaf_type is bool:
            return False
    if isinstance(x, (list, tuple)):
        return [_fill_none(v, leaf_type) for v in x]
    return x


def _to_tensor(x: Any, leaf_type: type) -> torch.Tensor:
    if leaf_type not in TORCH_DTYPE_MAP:
        raise TypeError(f"Unsupported leaf type: {leaf_type}")
    x = _fill_none(x, leaf_type)
    # 仅支持标量或嵌套 list/tuple；若是 dict 说明 config 不是叶子或数据格式不符
    if isinstance(x, Mapping):
        raise TypeError("Expected list/tuple/scalar at leaf, but got dict. Check config paths.")
    return torch.tensor(x, dtype=TORCH_DTYPE_MAP[leaf_type])


def materialize_record_to_tensors(record: Dict[str, Any],
                                  config: Dict[str, Any],
                                  base_path: List[str] = None) -> Dict[str, Any]:
    """
    从单条 record 中，按 config 的结构与叶子类型取值，并把叶子转为 torch.tensor。
    返回结构与 config 相同。
    """
    base_path = base_path or []
    out: Dict[str, Any] = {}
    for key, spec in config.items():
        if isinstance(spec, dict):
            out[key] = materialize_record_to_tensors(record, spec, base_path + [key])
        else:
            # 叶子：spec 是类型提示（float/int/bool），据此转 tensor
            value = _get_nested(record, base_path + [key])
            out[key] = _to_tensor(value, spec)
    return out


def materialize_input_leaf_tensors(
        data: Dict[str, Any],
        config: Dict[str, Any],
        base_path: List[str] = None,
        sep: str = '/') -> Dict[str, torch.Tensor]:
    """
    沿 config 的叶子路径，从 data 中取值并转为张量。
    返回形如 {'forward/values': Tensor, 'forward/masks': Tensor, ...} 的扁平字典。
    """
    base_path = base_path or []
    out: Dict[str, torch.Tensor] = {}
    for key, spec in config.items():
        if isinstance(spec, dict):
            out.update(materialize_input_leaf_tensors(data, spec, base_path + [key], sep=sep))
        else:
            leaf_path = base_path + [key]
            value = _get_nested(data, leaf_path)
            out[sep.join(leaf_path)] = _to_tensor(value, spec)
    return out
