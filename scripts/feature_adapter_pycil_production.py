#!/usr/bin/env python3
"""PyCIL production final-task feature adapter (FeTrIL, FeCAM, SSRE, PASS, EFC).

This is an inference-only adapter for scripts/feature_space_pipeline.py.
It restores the standardized network state_dict produced by
thesis.experiments.runner.save_network_checkpoint (NOT method_state pickle).

Important: the first real production checkpoint for each method MUST pass the
staged smoke audit before running full extraction. This file was constructed
from user-exported source snapshots; no real checkpoint weights were supplied.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

FEATURE_ADAPTER_VERSION = "1.1.0"


def _bootstrap_pycil_imports() -> Path:
    """Add the real PyCIL project root to sys.path for script-path invocations.

    `python scripts/feature_space_pipeline.py` puts scripts/ (not PyCIL/) at
    sys.path[0]. The loader imports this adapter by file path, so imports such
    as `thesis.data.loaders` fail unless we explicitly expose PyCIL/.
    This uses only directories that contain BOTH expected PyCIL modules.
    """
    adapter_file = Path(__file__).resolve()
    searched = []
    for root in adapter_file.parents:
        searched.append(str(root))
        if ((root / "thesis" / "data" / "loaders.py").is_file()
                and (root / "utils" / "data_manager.py").is_file()):
            root_str = str(root)
            # Put this repo ahead of unrelated installed packages named thesis.
            if root_str in sys.path:
                sys.path.remove(root_str)
            sys.path.insert(0, root_str)
            return root
    raise FileNotFoundError(
        "Could not locate the PyCIL source root containing both "
        "thesis/data/loaders.py and utils/data_manager.py. "
        "Expected adapter location: PyCIL/scripts/feature_adapter_pycil_production.py. "
        "Searched ancestor folders: " + ", ".join(searched)
    )


def _resolve_existing_path(value, root: Path, *fallbacks: Path) -> Path:
    candidates = []
    if value and '<PROJECT_ROOT>' not in str(value):
        original = Path(str(value))
        candidates.extend([original, root / original])
    candidates.extend(root / candidate for candidate in fallbacks)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError('Required data path not found; checked: ' + ', '.join(map(str, candidates)))


def _resolve_args(effective_args: dict, project_root: Path, protocol: str, device: str) -> dict:
    args = dict(effective_args)
    ds = str(args.get('dataset', '')).lower()
    if ds != 'aid':
        raise ValueError('This adapter currently supports AID production runs only')
    split_rel = Path('splits') / 'aid' / 'seed_1993'
    args['dataset_root'] = str(_resolve_existing_path(
        args.get('dataset_root'), project_root, Path('datasets/raw/AID')))
    args['split_root'] = str(_resolve_existing_path(
        args.get('split_root'), project_root, split_rel))
    args['task_config_path'] = str(_resolve_existing_path(
        args.get('task_config_path'), project_root, split_rel / f'{protocol}_tasks.json'))
    args['device'] = [device]
    args['num_workers'] = 0  # deterministic and Windows-safe; no training
    return args


def _network_for_method(method: str, args: dict, task_sizes: list[int]):
    # The implementations below follow learner constructors and incremental_train
    # network-head creation, but never invoke any training method or modify weights.
    from utils.inc_net import IncrementalNet, CosineIncrementalNet
    if method == 'efc':
        from models.efc import EFCNet
        network = EFCNet(args)
        for n in task_sizes:
            network.add_head(n)
    elif method == 'fecam':
        network = CosineIncrementalNet(args, False)
        seen = 0
        for task_id, n in enumerate(task_sizes):
            seen += n
            network.update_fc(seen, task_id)
    elif method in {'fetril', 'ssre', 'pass'}:
        network = IncrementalNet(args, False)
        count = sum(task_sizes) * (4 if method == 'pass' else 1)
        network.update_fc(count)
    else:
        raise ValueError(f'Unexpected method {method!r}')
    return network


def _checkpoint_state(path: Path, torch):
    # weights_only=True protects against arbitrary pickled Python objects.
    try:
        chk = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('This torch version does not support weights_only=True; upgrade torch to safely read the production network checkpoint') from exc
    if not isinstance(chk, dict) or not isinstance(chk.get('model_state_dict'), dict):
        raise ValueError('Network checkpoint must contain model_state_dict dictionary')
    state = chk['model_state_dict']
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('Checkpoint contains non-tensor or empty network state')
    if any(str(k).startswith('module.') for k in state):
        # Standard runner removes DataParallel, but allow older saves transparently.
        if not all(str(k).startswith('module.') for k in state):
            raise ValueError('Mixed DataParallel/non-DataParallel state_dict keys')
        state = {str(k)[7:]: v for k, v in state.items()}
    return chk, state


def _runtime_device(torch) -> str:
    request = os.getenv('FEATURE_DEVICE', 'auto').strip().lower()
    if request in {'auto', ''}:
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if request == 'cpu':
        return 'cpu'
    if request.startswith('cuda:'):
        ix = int(request.split(':', 1)[1])
        if not torch.cuda.is_available() or ix < 0 or ix >= torch.cuda.device_count():
            raise RuntimeError(f'FEATURE_DEVICE={request} is unavailable')
        return request
    raise ValueError('FEATURE_DEVICE must be auto, cpu, or cuda:<index>')


def extract_from_checkpoint(*, run_dir, checkpoint_path, effective_args,
                            method, protocol, backbone, attention, limit=None):
    """Return {features, y_true, sample_id, sample_path, checkpoint_loaded, semantic_eval_features}.

    No weights training; all features come from a final production network.
    Validation of y_true order against canonical prediction file is delegated to
    feature_space_pipeline.verify_extraction().
    """
    # Import the local `thesis` package from PyCIL/, not from scripts/ or site-packages.
    project_root = _bootstrap_pycil_imports()
    import torch
    from thesis.data.loaders import build_eval_loader
    from utils.data_manager import DataManager

    run_dir = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file() or run_dir not in checkpoint_path.parents:
        raise FileNotFoundError(f'Checkpoint missing or not inside the requested run: {checkpoint_path}')
    if limit is not None and int(limit) <= 0:
        raise ValueError('limit must be positive')

    method = str(method).lower()
    if method not in {'fetril', 'fecam', 'ssre', 'pass', 'efc'}:
        raise ValueError(f'Unsupported learner method: {method!r}')
    if protocol not in ('T5', 'T10'):
        raise ValueError('Unknown task protocol')
    device = _runtime_device(torch)
    args = _resolve_args(effective_args, project_root, protocol, device)
    # Metadata consistency is vital: never accidentally compare a different run.
    expected_name = f"aid_{method}_{protocol.lower()}_{backbone}_{attention}_seed{args['seed']}"
    if run_dir.name != expected_name:
        raise ValueError(f'Run metadata mismatch: {run_dir.name} != {expected_name}')
    if str(args['model_name']).lower() != method or str(args['attention_type']).lower() != attention:
        raise ValueError('effective_args model_name / attention_type conflicts with run metadata')
    required_convnet = ('remote_ssre_resnet' if method == 'ssre' else 'remote_resnet') + ('18' if backbone == 'r18' else '34')
    if args.get('convnet_type') != required_convnet:
        raise ValueError(f'Unexpected convnet_type {args.get("convnet_type")} (expected {required_convnet})')

    # DataManager follows production task config + manifests; eval uses no random transforms.
    dm = DataManager(args['dataset'], args['shuffle'], int(args['seed']), 0, 0,
                     dataset_root=args['dataset_root'], split_root=args['split_root'],
                     task_config_path=args['task_config_path'])
    task_sizes = [int(dm.get_task_size(i)) for i in range(dm.nb_tasks)]
    if len(task_sizes) != int(protocol[1:]) or sum(task_sizes) != 30:
        raise ValueError(f'Unexpected AID class progression: {task_sizes}')
    task_id = dm.nb_tasks - 1
    if checkpoint_path.name != f'model_{task_id}.pkl':
        raise ValueError(f'Checkpoint {checkpoint_path.name} is not final task {task_id}')
    if not (run_dir / 'model_checkpoints' / f'checkpoint_{task_id}.json').is_file():
        raise FileNotFoundError('Final task checkpoint manifest missing')

    network = _network_for_method(method, args, task_sizes)
    chk, state = _checkpoint_state(checkpoint_path, torch)
    if int(chk.get('tasks', -999)) != task_id:
        raise ValueError(f'Checkpoint task id mismatch: {chk.get("tasks")} != {task_id}')
    # strict=True ensures every key/shape matches, including attention and classifier architecture.
    network.load_state_dict(state, strict=True)
    if method == 'ssre':
        # SSRE merges adapter weights into conv weights at end of each task.
        # The final network therefore runs in normal mode, not parallel_adapters.
        network.convnet.switch('normal')
    network = network.to(torch.device(device)).eval()
    if int(network.feature_dim) != 512:
        raise ValueError(f'Unexpected feature_dim={network.feature_dim}; expected 512')

    dataset = dm.get_dataset(np.arange(30), source='test', mode='test')
    if len(dataset) != 2000:
        raise ValueError(f'AID final test set expected 2000 samples, found {len(dataset)}')
    # For contact sheets and sample provenance. The existing prediction .npz has no sample IDs,
    # so exact identity remains a manifest-derived claim unless a source ID artifact exists.
    ordered_paths = np.asarray(dataset.images, dtype=str)
    ordered_ids = np.asarray([str(Path(p).relative_to(args['dataset_root'])).replace('\\','/')
                              if Path(p).is_relative_to(args['dataset_root']) else str(p)
                              for p in ordered_paths], dtype=str)
    n = len(dataset) if limit is None else min(len(dataset), int(limit))
    batch_size = int(os.environ.get('FEATURE_BATCH_SIZE', '2'))
    if not 1 <= batch_size <= 32:
        raise ValueError('FEATURE_BATCH_SIZE must be between 1 and 32 for 600x600 AID images')
    loader = build_eval_loader(dataset, batch_size=batch_size, num_workers=0,
                               pin_memory=device.startswith('cuda'))
    all_features, all_labels = [], []
    with torch.inference_mode():
        for _, images, labels in loader:
            if sum(len(x) for x in all_labels) >= n:
                break
            images = images.to(device, non_blocking=device.startswith('cuda'))
            features = network.extract_vector(images)
            if not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != 512:
                raise ValueError(f'Invalid output from network.extract_vector: {getattr(features, "shape", None)}')
            remaining = n - sum(len(x) for x in all_labels)
            all_features.append(features[:remaining].detach().float().cpu().numpy())
            all_labels.append(labels[:remaining].numpy().astype(np.int64, copy=False))
    y = np.concatenate(all_labels)
    vectors = np.concatenate(all_features).astype(np.float32, copy=False)
    if vectors.shape != (n, 512) or len(y) != n:
        raise RuntimeError(f'Incorrect extracted feature shape {vectors.shape}, labels {y.shape}')
    if not np.array_equal(y, dataset.labels[:n]):
        raise RuntimeError('Evaluation dataloader modified test-sample ordering')
    return {
        'checkpoint_loaded': True,
        'semantic_eval_features': True,
        'features': vectors,
        'y_true': y,
        'sample_id': ordered_ids[:n],
        'sample_path': ordered_paths[:n],
    }
