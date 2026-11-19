#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export PyCIL feature-loader context without dataset, checkpoint weights or raw predictions.

Run at the PyCIL root (Windows or Linux):
  python scripts/export_feature_adapter_context.py --machine pc2
  python scripts/export_feature_adapter_context.py --machine server

Security: JSON keys containing common secret names are redacted; Python sources are
restricted to explicit code directories and suspicious credential assignments are
excluded. Review the ZIP before sharing. Never loads model/method state by default.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUN_RE = re.compile(
    r'^(aid|nwpu)_(fetril|fecam|ssre|pass|efc)_(t5|t10)_(r18|r34)_(none|eca|cbam|procbam)_seed(\d+)$',
    re.IGNORECASE)
METHODS = ('fetril', 'fecam', 'ssre', 'pass', 'efc')
BAD_KEY_RE = re.compile(
    r'(?i)(password|passwd|secret|api[_-]?key|access[_-]?key|private[_-]?key|credential|auth[_-]?token|bearer|cookie|webhook|client[_-]?secret|dsn|connection[_-]?string|smtp[_-]?password|login[_-]?token)')
BAD_SOURCE_NAME_RE = re.compile(r'(?i)(\.env|secret|credential|token|password|keypair|private[_-]?key|id_rsa)')
POTENTIAL_SECRET_ASSIGNMENT = re.compile(
    r'''(?im)^\s*[\w\.\[\]"']*(?:API_KEY|PASSWORD|SECRET|ACCESS_TOKEN|PRIVATE_KEY|AUTH_TOKEN|GITHUB_TOKEN)\s*[\w\.\[\]"']*\s*[:=]\s*["']([^"'\n]{8,})["']''')
MAX_JSON_BYTES = 512_000
MAX_SOURCE_FILE_BYTES = 512_000
MAX_TOTAL_SOURCE_BYTES = 7_000_000


def json_dumps(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sanitize_text(text: str, root: Path) -> str:
    """Anonymize well-known absolute paths, not contents of experiment identifiers."""
    s = str(text)
    for anchor, replacement in ((str(root), '<PROJECT_ROOT>'), (str(Path.home()), '<HOME>')):
        if anchor:
            s = s.replace(anchor, replacement).replace(anchor.replace('\\', '/'), replacement)
    s = re.sub(r'(?i)C:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+', '<USER_HOME>', s)
    s = re.sub(r'/home/[^/\s]+', '<USER_HOME>', s)
    return s


def scrub_json(value, root: Path, stats: Counter, level=0):
    if level > 28:
        stats['depth_limited'] += 1
        return '<MAX_DEPTH>'
    if isinstance(value, dict):
        result = {}
        for key, val in value.items():
            k = str(key)
            if BAD_KEY_RE.search(k):
                result[k] = '<REDACTED>'
                stats['redacted_keys'] += 1
            else:
                result[k] = scrub_json(val, root, stats, level+1)
        return result
    if isinstance(value, (list, tuple)):
        return [scrub_json(v, root, stats, level+1) for v in value[:10000]]
    if isinstance(value, str):
        s = sanitize_text(value, root)
        # Credentials sometimes occur inside URLs even when the key name looks harmless.
        s = re.sub(r'(?i)(https?://)[^/@\s:]+:[^/@\s]+@', r'\1<AUTH>@', s)
        s = re.sub(r'(?i)((?:api[_-]?key|password|token|secret)\s*[=:]\s*)\S+', r'\1<REDACTED>', s)
        return s[:25000] if len(s) > 25000 else s
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    stats['unknown_types'] += 1
    return '<UNSUPPORTED_TYPE>'


def read_json_sanitized(path: Path, root: Path, stats: Counter):
    if not path.is_file():
        return None, 'missing'
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None, 'file_too_large'
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return scrub_json(value, root, stats), None
    except (OSError, ValueError, UnicodeError) as e:
        return None, f'{type(e).__name__}'


def get_package_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn", "ultralytics"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def shallow_git_commit(root: Path):
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                           text=True, timeout=5, check=False)
        sha = r.stdout.strip()
        return sha if r.returncode == 0 and re.fullmatch('[a-f0-9]{40}', sha) else None
    except Exception:
        return None


def get_prediction_schema(path: Path):
    out = {'exists': path.is_file()}
    if not path.is_file():
        return out
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            out['arrays'] = {name: {'shape': list(z[name].shape), 'dtype': str(z[name].dtype)}
                             for name in z.files}
            if 'y_true' in z.files:
                y = z['y_true']
                out['y_true_order_sha256'] = hash_bytes(y.tobytes())
                out['y_true_n'] = int(y.size)
                labels, counts = np.unique(y, return_counts=True)
                out['y_true_histogram'] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
    except Exception as e:
        out['inspection_error'] = type(e).__name__
    return out


def find_source_files(root: Path):
    targets = []
    for d in ('models', 'convs', 'utils', 'thesis', 'datasets'):
        base = root / d
        if base.is_dir():
            targets += [p for p in base.rglob('*.py') if p.is_file()]
    for name in ('main.py', 'trainer.py', 'factory.py', 'requirements.txt',
                 'pyproject.toml', 'setup.py', 'scripts/production_section.py',
                 'scripts/feature_space_pipeline.py', 'scripts/feature_adapter_template.py',
                 'scripts/retry_incomplete_section.py'):
        p = root / name
        if p.is_file():
            targets.append(p)
    return sorted(set(targets))


def collect_source(root: Path, stats: Counter):
    payloads = {}
    excluded = []
    total = 0
    for p in find_source_files(root):
        rel = p.relative_to(root).as_posix()
        if any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(root).parts) or BAD_SOURCE_NAME_RE.search(p.name):
            excluded.append({'file': rel, 'reason': 'excluded_filename'})
            continue
        try:
            size = p.stat().st_size
            if size > MAX_SOURCE_FILE_BYTES:
                excluded.append({'file': rel, 'reason': 'source_too_large'})
                continue
            if total + size > MAX_TOTAL_SOURCE_BYTES:
                excluded.append({'file': rel, 'reason': 'total_source_limit'})
                continue
            text = p.read_text(encoding='utf-8-sig')
            if POTENTIAL_SECRET_ASSIGNMENT.search(text):
                excluded.append({'file': rel, 'reason': 'credential_literal_detected_manual_review_required'})
                continue
            # Preserve code structure, anonymize paths only.
            b = sanitize_text(text, root).encode('utf-8')
            payloads[f'source/{rel}'] = b
            total += size
        except Exception as e:
            excluded.append({'file': rel, 'reason': type(e).__name__})
    stats['source_files'] = len(payloads)
    stats['source_excluded'] = len(excluded)
    return payloads, excluded


def weight_structure_smoke(path: Path):
    """Optionally inspect ONE checkpoint per method with restricted PyTorch unpickling.
    Does not extract embeddings, does not prove learner loadability.
    """
    result = {'weights_only_attempted': True, 'file': path.name, 'exists': path.is_file()}
    if not path.is_file():
        return result
    try:
        import torch
        obj = torch.load(str(path), map_location='cpu', weights_only=True)
        result['weights_only_readable'] = True
        result['top_level_type'] = type(obj).__name__
        if isinstance(obj, dict):
            result['top_level_keys'] = [str(k)[:128] for k in list(obj.keys())[:100]]
            tensors = []
            # Only a shallow inspection; do not include actual values.
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    tensors.append({'key': str(k)[:128], 'shape': list(v.shape), 'dtype': str(v.dtype)})
                elif isinstance(v, dict):
                    for subk, subv in list(v.items())[:500]:
                        if isinstance(subv, torch.Tensor):
                            tensors.append({'key': str(k)[:64]+'.'+str(subk)[:128], 'shape': list(subv.shape), 'dtype': str(subv.dtype)})
            result['tensor_count_inspected'] = len(tensors)
            result['tensor_schema_first_120'] = tensors[:120]
        del obj
    except Exception as e:
        result['weights_only_readable'] = False
        result['error_type'] = type(e).__name__
        result['error_message'] = str(e)[:500]
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--machine', choices=('pc2', 'server'), required=True)
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--source', default='outputs/experiments')
    ap.add_argument('--output-dir', default='outputs/feature_analysis/context_exports')
    ap.add_argument('--mode', choices=('all', 'smoke'), default='all',
                    help='Export all lightweight JSON for every run, or only one baseline per method.')
    ap.add_argument('--without-source', action='store_true', help='Do not bundle allowlisted Python source')
    ap.add_argument('--inspect-network-structure', action='store_true',
                    help='Optionally inspect one model pkl per method using torch.load(weights_only=True), CPU only')
    args = ap.parse_args(argv)
    root = Path(args.project_root).resolve()
    src = (root / args.source).resolve()
    output = (root / args.output_dir).resolve()
    if not src.is_dir():
        ap.error(f'Experiment folder not found: {src}')
    output.mkdir(parents=True, exist_ok=True)

    candidates = []
    for run in sorted(src.iterdir()):
        if run.is_dir():
            mat = RUN_RE.fullmatch(run.name)
            if mat:
                ds, method, proto, backbone, attn, seed = mat.groups()
                if ds.lower() != 'aid':
                    continue
                candidates.append({'path': run, 'dataset':ds.lower(), 'method':method.lower(),
                                   'protocol':proto.upper(), 'backbone':backbone.lower(),
                                   'attention':attn.lower(), 'seed':int(seed), 'run_id':run.name})
    candidates.sort(key=lambda r:(r['method'],r['attention'] != 'none',r['protocol'] != 'T5',r['backbone']!='r18',r['run_id']))
    smoke_runs = {}
    for run in candidates:
        smoke_runs.setdefault(run['method'], run)
    selected = candidates if args.mode == 'all' else list(smoke_runs.values())

    archive_data = {}
    stats = Counter()
    rows = []
    warnings = []
    for r in selected:
        run = r['path']
        task_id = 4 if r['protocol'] == 'T5' else 9
        prefix = 'run_metadata/' + r['run_id']
        file_map = {
            'effective_args.json': run/'effective_args.json',
            'config.json': run/'config.json',
            'provenance.json': run/'provenance.json',
            'status.json': run/'status.json',
            f'checkpoint_{task_id}.json': run/'model_checkpoints'/f'checkpoint_{task_id}.json',
        }
        missing = []
        for short_name, path in file_map.items():
            doc, reason = read_json_sanitized(path, root, stats)
            if reason is None:
                archive_data[prefix+'/'+short_name] = json_dumps(doc)
                stats['metadata_json_files'] += 1
            else:
                missing.append(short_name+':'+reason)
        ckpt = run/'model_checkpoints'/f'model_{task_id}.pkl'
        method_state = run/'model_checkpoints'/f'method_state_{task_id}.pkl'
        pred = run/'predictions'/f'task_{task_id:02d}.npz'
        row = {k:r[k] for k in ('run_id','dataset','method','protocol','backbone','attention','seed')}
        row.update({'machine':args.machine, 'final_task':task_id,
                    'checkpoint_model_exists':ckpt.is_file(),
                    'checkpoint_model_bytes':ckpt.stat().st_size if ckpt.is_file() else 0,
                    'checkpoint_method_state_exists':method_state.is_file(),
                    'checkpoint_method_state_bytes':method_state.stat().st_size if method_state.is_file() else 0,
                    'metadata_missing':missing,
                    'prediction_schema':get_prediction_schema(pred)})
        rows.append(row)
        if missing:
            warnings.append({'run_id':r['run_id'],'missing':missing})
    if not args.without_source:
        source_files, excluded = collect_source(root, stats)
        archive_data.update(source_files)
        archive_data['source_exclusions.json'] = json_dumps(excluded)
    smoke_summary = {}
    if args.inspect_network_structure:
        for method, r in smoke_runs.items():
            task_id = 4 if r['protocol'] == 'T5' else 9
            p = r['path'] / 'model_checkpoints' / f'model_{task_id}.pkl'
            smoke_summary[method] = {'run_id': r['run_id'], **weight_structure_smoke(p)}
        archive_data['network_weight_structure_smoke.json'] = json_dumps(scrub_json(smoke_summary,root,stats))

    machine_summary = {'schema_version':'1.0', 'generated_at':datetime.now(timezone.utc).isoformat(),
        'machine':args.machine, 'git_commit':shallow_git_commit(root),
        'package_versions':get_package_versions(),
        'mode':args.mode, 'total_local_aid_runs_found':len(candidates),
        'runs_exported':len(rows), 'methods_local':dict(Counter(r['method'] for r in candidates)),
        'smoke_candidates':{method:r['run_id'] for method,r in smoke_runs.items()},
        'metadata_export':{'source_files':int(stats['source_files']), 'metadata_json_files':int(stats['metadata_json_files']),
                           'keys_redacted':int(stats['redacted_keys']),'source_excluded':int(stats['source_excluded'])},
        'warnings':warnings,
        'notes':['No datasets, raw predictions or checkpoint pkl are included.',
                 'Prediction y_true hashes describe canonical label ORDER only; raw labels are not exported.',
                 'Checkpoint presence/weights-only schema do not prove restore or feature extractability.',
                 'Inspect ZIP manually before sharing to ensure any project-specific sensitive code is absent.']}
    archive_data['manifest.json'] = json_dumps(machine_summary)
    archive_data['run_inventory.json'] = json_dumps(rows)
    archive_data['README.txt'] = (f'PyCIL feature adapter context export — {args.machine}\n'
                                  'Only source Python, sanitized JSON configs/manifests, prediction shapes and label hashes.\n'
                                  'No .pkl, .npz, datasets, .env, passwords or model weights are packaged.\n'
                                  'Review the ZIP before sharing.\n').encode('utf-8')
    name = f'feature_adapter_context_{args.machine}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    destination = output/name
    temp = output/(name+'.partial')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zf:
        for name,content in sorted(archive_data.items()):
            zf.writestr(name,content)
    with zipfile.ZipFile(temp,'r') as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError('Zip CRC failed: '+bad)
    temp.replace(destination)
    print(json.dumps({'machine':args.machine,'runs_scanned':len(candidates),'runs_exported':len(rows),
        'source_files':int(stats['source_files']), 'redacted_json_keys':int(stats['redacted_keys']),
        'output_zip':str(destination),'zip_bytes':destination.stat().st_size,
        'excluded_sensitive_source_files':int(stats['source_excluded'])},indent=2))
    return destination


if __name__ == '__main__':
    main()
