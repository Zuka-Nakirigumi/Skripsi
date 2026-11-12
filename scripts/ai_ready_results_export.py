#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit PyCIL production and create AI-ready evidence exports (no external LLM/API).

Run from PyCIL root or import through notebooks/08_ai_ready_insights.ipynb.
Requires numpy and pandas. Does not load model weights or alter training files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# Notebook/exporter compatibility marker. Do not remove without updating notebook.
AI_READY_EXPORTER_VERSION = "2.1.0"
REQUIRED_ALL_MODELS_OUTPUTS = (
    "model_insights_all.csv",
    "bundle_insights_all.csv",
    "attention_insights_all.csv",
    "class_insights_final_all.csv",
    "insights_all.json",
    "AI_EVIDENCE_REPORT.md",
)

ATTENTIONS = ("none", "eca", "cbam", "procbam")
METHODS = ("fetril", "fecam", "ssre", "efc", "pass")
BACKBONES = ("r18", "r34")
PROTOCOLS = ("T5", "T10")
ALLOW_CHECKPOINT_INVENTORY = True
EXPECTED_SEED = 1993
SOURCE_TABLES = {
    'run_audit.csv': 'Status eligibility tiap run, termasuk reason jika tidak layak.',
    'expected_coverage.csv': 'Grid AID/NWPU yang seharusnya ada, status tiap kombinasi.',
    'valid_runs.csv': 'Metrik canonical untuk run yang terverifikasi.',
    'paired_attention_effects.csv': 'Peningkatan/penurunan relative terhadap baseline None yang dipasangkan secara fair.',
    'rejected_pairs.csv': 'Attention pair yang ditolak karena konfigurasi/data evaluasi tidak sebanding.',
    'method_attention_gain.csv': 'Rata-rata delta AIA per method dan attention beserta n pasangan dan coverage.',
    'task_accuracy_long.csv': 'Joint accuracy untuk setiap incremental task.',
    'class_accuracy_long.csv': 'Per-class recall/accuracy per task, termasuk sample support.',
    'class_forgetting.csv': 'Puncak akurasi kelas sejak diperkenalkan minus akurasi akhir.',
    'class_attention_delta.csv': 'Δ class accuracy untuk attention vs None, tiap task dan kelas.',
    'class_attention_final_wide.csv': 'Perbandingan empat attention pada task akhir, satu baris per kelas.',
    'task_attention_delta.csv': 'Perubahan joint/old/new accuracy pada task yang sama vs baseline.',
    'class_forgetting_attention_delta.csv': 'Perubahan forgetting per kelas vs baseline.',
    'misclassification_long.csv': 'Arah salah klasifikasi aktual pada task akhir (true -> predicted).',
    'misclassification_delta.csv': 'Perubahan error count dan row-normalized error-rate untuk pasangan kelas.',
    'class_label_map.csv': 'Pemetaan incremental ID ke nama kelas atau fallback ID tanpa dugaan nama.',
}
CLASS_ORDER = {
    'aid': [27, 21, 26, 2, 8, 13, 10, 6, 15, 20, 14, 23, 12, 28, 11, 5, 0, 22, 18, 4, 19, 24, 9, 7, 16, 3, 29, 17, 25, 1],
    'nwpu': [21, 35, 13, 17, 25, 18, 10, 22, 41, 8, 2, 38, 27, 1, 16, 37, 0, 30, 43, 12, 6, 15, 20, 24, 19, 14, 34, 26, 36, 23, 7, 40, 28, 11, 5, 32, 4, 9, 42, 39, 31, 3, 29, 44, 33],
}
METADATA_COLS = ['dataset','method','protocol','backbone','attention','seed','batch','source_machine','run_id']
KEYS = ['dataset', 'method', 'protocol', 'backbone', 'seed']
PAIR_KEYS = KEYS + ['source_machine','batch']

def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def numeric(value):
    if isinstance(value, bool) or value is None:
        return np.nan
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def safe_float(value):
    value = numeric(value)
    return None if pd.isna(value) else float(value)


def select_value(documents, keys):
    """Ambil nilai dari config/args/provenance tanpa menebak bila tidak ada."""
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        for key in keys:
            if key in doc and doc[key] is not None and not isinstance(doc[key], (dict, list)):
                return doc[key]
        for container in ("config", "args", "experiment", "training", "run_config"):
            sub = doc.get(container)
            if isinstance(sub, dict):
                for key in keys:
                    if key in sub and sub[key] is not None and not isinstance(sub[key], (dict, list)):
                        return sub[key]
    return None


def normalize_metadata(value, kind):
    if value is None:
        return None
    s = str(value).strip().lower().replace("-", "_")
    if kind == "dataset":
        if "resisc" in s or "nwpu" in s: return "nwpu"
        if s in ("aid", "aid30", "aid_dataset"): return "aid"
    if kind == "method":
        for m in METHODS:
            if s == m or s == f"{m}_learner": return m
    if kind == "protocol":
        if s in ("5", "t5", "t_5"): return "T5"
        if s in ("10", "t10", "t_10"): return "T10"
    if kind == "backbone":
        if re.search(r"(?:^|_)r18(?:_|$)", s) or "resnet18" in s or "resnet_18" in s: return "r18"
        if re.search(r"(?:^|_)r34(?:_|$)", s) or "resnet34" in s or "resnet_34" in s: return "r34"
    if kind == "attention":
        if s in ("none", "baseline", "no_attention", "no_attn", "null"): return "none"
        for a in ("procbam", "cbam", "eca"):
            if re.search(r"(?:^|_)" + a + r"(?:_|$)", s): return a
        # Hanya infer none jika field attention sendiri explicit; jangan dari 'r18'.
    if kind == "seed":
        try: return int(value)
        except (ValueError, TypeError): return None
    if kind == "batch":
        try: return int(value)
        except (ValueError, TypeError): return None
    return None


def infer_from_run_name(run_id, kind):
    s = str(run_id).lower().replace("-", "_")
    if kind == "method":
        for m in METHODS:
            if re.search(r"(?:^|_)" + m + r"(?:_|$)", s): return m
    if kind == "dataset":
        for token, normalized in (("aid", "aid"), ("nwpu", "nwpu"), ("resisc45", "nwpu")):
            if re.search(r"(?:^|_)" + token + r"(?:_|$)", s): return normalized
    if kind == "protocol":
        m = re.search(r"(?:^|_)t(5|10)(?:_|$)", s)
        if m: return "T" + m.group(1)
    if kind == "backbone":
        m = re.search(r"(?:^|_)(r18|r34)(?:_|$)", s)
        if m: return m.group(1)
    if kind == "attention":
        for a in ATTENTIONS:
            if re.search(r"(?:^|_)" + a + r"(?:_|$)", s): return a
    if kind == "seed":
        m = re.search(r"(?:^|_)seed_?(\d+)(?:_|$)", s)
        if m: return int(m.group(1))
    return None


def metadata_from_files(run_id, config, args, provenance, summary):
    docs = [config, args, provenance, summary]
    names = {
        "dataset": ["dataset", "dataset_name"],
        "method": ["method", "model_name", "learner_name"],
        "protocol": ["protocol", "task_protocol", "task_setting", "scenario"],
        "backbone": ["backbone", "backbone_type", "backbone_name", "convnet_type"],
        "attention": ["attention", "attention_type", "attention_name", "attn_type"],
        "seed": ["training_seed", "seed", "random_seed"],
        "batch": ["batch_size", "batch"],
    }
    result = {}
    for kind, aliases in names.items():
        raw = select_value(docs, aliases)
        val = normalize_metadata(raw, kind)
        if val is None and kind == "attention":
            # convnet_type/backbone id kadang berupa r34_procbam.
            val = normalize_metadata(select_value(docs, ["convnet_type", "backbone"]), "attention")
        if val is None:
            val = infer_from_run_name(run_id, kind)
        result[kind] = val
    return result


def canonical_primary_top1(primary):
    if not isinstance(primary, dict): return None
    if primary.get("top1") is not None: return safe_float(primary["top1"])
    metric = primary.get("metrics")
    if isinstance(metric, dict): return safe_float(metric.get("top1"))
    return None


def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred = cm.sum(axis=0)
    true = cm.sum(axis=1)
    denom = pred + true
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1) * 100.0)


def checkpoint_check(run_dir, n_tasks, allow_inventory):
    """Hanya structural check. Tidak memanggil torch.load/pickle.load."""
    ckpt = run_dir / "model_checkpoints"
    inventory_path = run_dir / "checkpoint_inventory.json"
    inv = None
    if allow_inventory and inventory_path.exists():
        inv = read_json(inventory_path).get("inventory")
    full, snapshot = True, isinstance(inv, dict)
    for task_id in range(n_tasks):
        for file_name in (f"model_{task_id}.pkl", f"method_state_{task_id}.pkl"):
            rel = "model_checkpoints/" + file_name
            actual = ckpt / file_name
            exists_full = actual.is_file() and actual.stat().st_size > 0
            full = full and exists_full
            item = inv.get(rel, {}) if isinstance(inv, dict) else {}
            snapshot = snapshot and (
                (isinstance(item, dict) and item.get("exists") is True
                 and isinstance(item.get("size"), int) and item["size"] > 0)
                or exists_full
            )
        manifest_file = ckpt / f"checkpoint_{task_id}.json"
        if not manifest_file.is_file():
            raise ValueError(f"Missing checkpoint manifest: task {task_id}")
        manifest = read_json(manifest_file)
        if not isinstance(manifest, dict) or not {"network_checkpoint", "method_state"}.issubset(manifest):
            raise ValueError(f"Invalid checkpoint manifest: task {task_id}")
    if full: return "full"
    if snapshot: return "snapshot"
    raise ValueError("Missing model/method-state checkpoint (or source inventory)")

EXPECTED_SEEN_CLASSES = {
    ("aid", "T5"): [14, 18, 22, 26, 30],
    ("aid", "T10"): [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    ("nwpu", "T5"): [25, 30, 35, 40, 45],
    ("nwpu", "T10"): [18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
}

REQUIRED_CONTINUAL = ("aia", "last_task_accuracy", "forgetting_rate", "bwt",
                      "macro_f1_final", "task_accuracy_matrix", "class_accuracy_matrix")


def audit_run(run_dir: Path, source_machine: str):
    record = {
        "source_machine": source_machine,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": "incomplete",
        "issue": "",
        "checkpoint_level": "none",
        "eligible": False,
    }
    try:
        for fname in ("config.json", "effective_args.json", "provenance.json",
                      "summary.json", "status.json", "continual_metrics.json", "task_metrics.json"):
            if not (run_dir / fname).is_file():
                raise ValueError(f"Missing {fname}")

        config, args, provenance, summary, status, cmtr, task_metrics = (
            read_json(run_dir / fname) for fname in
            ("config.json", "effective_args.json", "provenance.json",
             "summary.json", "status.json", "continual_metrics.json", "task_metrics.json")
        )
        meta = metadata_from_files(run_dir.name, config, args, provenance, summary)
        record.update(meta)
        record["reported_state"] = status.get("state") if isinstance(status, dict) else None

        if status.get("state") != "completed":
            raise ValueError(f"status.json state = {status.get('state')!r}")
        if not isinstance(args, dict) or args.get("save_checkpoints") is not True:
            raise ValueError("effective_args.save_checkpoints != true")
        for k in ("dataset", "method", "protocol", "backbone", "attention", "seed", "batch"):
            if meta[k] is None:
                raise ValueError(f"Metadata '{k}' unknown; inspect config.json/effective_args.json")
        if not isinstance(task_metrics, list):
            raise ValueError("task_metrics.json must be a list")
        n_tasks = 5 if meta["protocol"] == "T5" else 10
        if len(task_metrics) != n_tasks:
            raise ValueError(f"Expected {n_tasks} task metrics, found {len(task_metrics)}")
        if not isinstance(cmtr, dict) or any(key not in cmtr for key in REQUIRED_CONTINUAL):
            missing = [key for key in REQUIRED_CONTINUAL if not isinstance(cmtr, dict) or key not in cmtr]
            raise ValueError(f"Missing continual metrics: {missing}")
        curve = []
        seen = []
        last_cm = None
        for tid, metric in enumerate(task_metrics):
            if not isinstance(metric, dict): raise ValueError(f"Invalid task_metrics[{tid}]")
            pred_path = run_dir / "predictions" / f"task_{tid:02d}.npz"
            cm_path = run_dir / "confusion_matrices" / f"task_{tid:02d}.npy"
            if not pred_path.is_file() or not cm_path.is_file():
                raise ValueError(f"Missing predictions/confusion_matrices for task {tid}")
            with np.load(pred_path, allow_pickle=False) as pred:
                if not {"y_true", "y_pred_topk"}.issubset(pred.files):
                    raise ValueError(f"Invalid prediction fields on task {tid}")
                y_true = np.asarray(pred["y_true"])
                y_topk = np.asarray(pred["y_pred_topk"])
            if y_true.ndim != 1 or y_topk.ndim not in (1, 2):
                raise ValueError(f"Invalid prediction dimensions on task {tid}")
            pred_top1 = y_topk[:, 0] if y_topk.ndim == 2 else y_topk
            if len(pred_top1) != len(y_true) or len(y_true) == 0:
                raise ValueError(f"Prediction length mismatch task {tid}")
            seen_classes = int(metric["seen_classes"])
            cm = np.load(cm_path, allow_pickle=False)
            if cm.shape != (seen_classes, seen_classes):
                raise ValueError(f"Confusion shape mismatch task {tid}: {cm.shape}")
            if not np.isfinite(cm).all() or (cm < 0).any():
                raise ValueError(f"Invalid confusion counts task {tid}")
            if int(cm.sum()) != len(y_true):
                raise ValueError(f"Confusion sample count mismatch task {tid}")
            if not (np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(pred_top1.dtype, np.integer)):
                raise ValueError(f"Prediction labels must be integer on task {tid}")
            if ((y_true < 0) | (y_true >= seen_classes)).any() or ((pred_top1 < 0) | (pred_top1 >= seen_classes)).any():
                raise ValueError(f"Prediction label out of range task {tid}")
            actual_cm = np.zeros_like(cm, dtype=np.int64)
            np.add.at(actual_cm, (y_true, pred_top1), 1)
            if not np.array_equal(cm.astype(np.int64), actual_cm):
                raise ValueError(f"Confusion disagrees with canonical predictions task {tid}")
            canonical_top1 = safe_float(metric.get("canonical_top1"))
            calc_top1 = float(np.mean(y_true == pred_top1) * 100)
            if canonical_top1 is None or abs(canonical_top1 - calc_top1) > 0.051:
                raise ValueError(f"Canonical Top-1 mismatch task {tid}")
            primary = canonical_primary_top1(metric.get("primary"))
            if primary is not None and abs(primary - calc_top1) > 0.051:
                raise ValueError(f"Primary Top-1 mismatch task {tid}")
            curve.append(calc_top1)
            seen.append(seen_classes)
            last_cm = cm

        expected = EXPECTED_SEEN_CLASSES.get((meta["dataset"], meta["protocol"]))
        if expected is not None and seen != expected:
            raise ValueError(f"Seen class progression {seen}, expected {expected}")
        final_cm = np.load(run_dir / "final_confusion_matrix.npy", allow_pickle=False)
        if not np.array_equal(final_cm, last_cm):
            raise ValueError("Final confusion matrix differs from last task")
        aia, lta = safe_float(cmtr.get("aia")), safe_float(cmtr.get("last_task_accuracy"))
        if aia is None or lta is None or not (0 <= aia <= 100 and 0 <= lta <= 100):
            raise ValueError("Missing/invalid AIA or LTA")
        if abs(aia - float(np.mean(curve))) > 0.051 or abs(lta - curve[-1]) > 0.051:
            raise ValueError("AIA/LTA do not agree with per-task canonical predictions")
        macro_f1 = macro_f1_from_confusion(final_cm)
        stored_macro = safe_float(cmtr.get("macro_f1_final"))
        if stored_macro is not None:
            # Runner dapat menyimpan F1 0..1 atau dalam persen; cek keduanya.
            if min(abs(stored_macro - macro_f1), abs(stored_macro * 100 - macro_f1)) > 0.15:
                raise ValueError("Macro-F1 disagrees with final confusion matrix")
        checkpoint_level = checkpoint_check(run_dir, n_tasks, ALLOW_CHECKPOINT_INVENTORY)
        record.update({
            "status": "ready_full" if checkpoint_level == "full" else "ready_snapshot",
            "checkpoint_level": checkpoint_level,
            "eligible": True,
            "n_tasks": n_tasks,
            "n_classes_final": seen[-1],
            "aia": aia,
            "lta": lta,
            "macro_f1_pct": macro_f1,
            "fr": safe_float(cmtr.get("forgetting_rate")),
            "bwt": safe_float(cmtr.get("bwt")),
            "fwt_status": cmtr.get("fwt_status", "not_specified"),
            "task_curve": curve,
            "seen_classes_curve": seen,
        })
    except Exception as exc:
        record["issue"] = f"{type(exc).__name__}: {exc}"
        if record.get("reported_state") == "failed": record["status"] = "failed"
        elif record.get("reported_state") == "running": record["status"] = "running"
        else: record["status"] = "incomplete"
    return record


def csv_write(df: pd.DataFrame, path: Path, columns: list[str]):
    if df is None or df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding='utf-8-sig')
        return
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: json.dumps(x,ensure_ascii=False) if isinstance(x,(list,tuple,dict)) else x)
    out.to_csv(path, index=False, encoding='utf-8-sig')


def jwrite(path: Path, data):
    def clean(obj):
        if isinstance(obj, dict): return {str(k):clean(v) for k,v in obj.items()}
        if isinstance(obj, (tuple,list)): return [clean(x) for x in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return clean(float(obj))
        if isinstance(obj, float) and not np.isfinite(obj): return None
        if isinstance(obj, Path): return str(obj)
        return obj
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2), encoding='utf-8')


def scan_sources(sources: dict[str,Path], dataset_scope='aid'):
    records=[]
    for machine, root in sources.items():
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or not any((p/f).exists() for f in ('status.json','config.json','task_metrics.json')):
                continue
            records.append(audit_run(p,machine))
    runs=pd.DataFrame(records)
    if runs.empty:
        return runs, runs.copy()
    runs['selection_note']=''
    key_cols=KEYS+['attention']
    for key, group in runs[runs.eligible == True].groupby(key_cols,dropna=False):
        ids=list(group.index)
        if len(ids)<=1:continue
        r0=runs.loc[ids[0]]
        ref_cm=np.load(Path(r0.run_dir)/'final_confusion_matrix.npy',allow_pickle=False)
        same=all(
            abs(float(runs.at[i,'aia'])-float(r0.aia))<1e-7
            and abs(float(runs.at[i,'lta'])-float(r0.lta))<1e-7
            and np.allclose(runs.at[i,'task_curve'],r0.task_curve,atol=1e-7)
            and np.array_equal(np.load(Path(runs.at[i,'run_dir'])/'final_confusion_matrix.npy',allow_pickle=False),ref_cm)
            and runs.at[i,'batch']==r0.batch
            and runs.at[i,'source_machine']==r0.source_machine
            for i in ids[1:])
        if not same:
            for i in ids:
                runs.at[i,'eligible']=False
                runs.at[i,'status']='duplicate_conflict'
                runs.at[i,'selection_note']='Same experiment key, different result/batch/machine; manual resolution required'
        else:
            chosen=sorted(ids,key=lambda i:(runs.at[i,'checkpoint_level']=='full',str(runs.at[i,'run_dir'])),reverse=True)[0]
            for i in ids:
                if i!=chosen:
                    runs.at[i,'eligible']=False
                    runs.at[i,'status']='duplicate_copy'
                    runs.at[i,'selection_note']='Identical duplicate of '+str(runs.at[chosen,'run_id'])
    valid=runs.loc[runs.eligible == True].copy()
    if dataset_scope not in (None,'all') and not valid.empty:
        valid=valid[valid.dataset == dataset_scope].copy()
    return runs,valid


def expected_coverage(runs, valid, scope='aid', seed=1993):
    datasets=['aid','nwpu'] if scope in (None,'all') else [scope]
    rows=[]
    for dataset, method, protocol, backbone, attention in product(datasets,METHODS,PROTOCOLS,BACKBONES,ATTENTIONS):
        assigned=('pc2' if method in ('pass','fecam','ssre') and backbone == 'r18' else 'server') if dataset=='aid' else 'unassigned'
        rows.append({'dataset':dataset,'method':method,'protocol':protocol,'backbone':backbone,'attention':attention,'seed':seed,'expected_machine':assigned})
    grid=pd.DataFrame(rows)
    keys=KEYS+['attention']
    if not valid.empty:
        found=valid[keys+['status','source_machine','batch','run_id']].copy()
        grid=grid.merge(found,on=keys,how='left',validate='one_to_one')
    else:
        grid['status']=np.nan
    grid['coverage']=grid.status.fillna('missing')
    grid['ready']=grid.coverage.isin(['ready_full','ready_snapshot'])
    # Include candidate non-ready statuses to diagnose failed/incomplete sections.
    if not runs.empty:
        other=runs[runs.eligible != True].copy()
        other=other.dropna(subset=keys)
        if not other.empty:
            lookup=(other.groupby(keys,dropna=False).status.apply(lambda s:';'.join(sorted(set(s)))).reset_index(name='non_ready_status'))
            grid=grid.merge(lookup,on=keys,how='left')
    return grid


def load_class_labels(project_root:Path,datasets:list[str],override:Path|None=None):
    """Never guess a class label. Explicit CSV wins; otherwise use verified original ID/name manifest.

    Expected explicit CSV: dataset,incremental_id,class_name.
    Manifest auto-map only when original_class_id + class_name exist, and fixed order seed 1993.
    """
    overrides={}
    csv_candidates=[]
    if override is not None: csv_candidates.append(Path(override))
    csv_candidates.append(project_root/'analysis_inputs'/'class_labels.csv')
    for candidate in csv_candidates:
        if candidate.is_file():
            df=pd.read_csv(candidate)
            if not {'dataset','incremental_id','class_name'}.issubset(df.columns):
                raise ValueError(f'{candidate} needs dataset,incremental_id,class_name')
            for _,rec in df.iterrows():
                ds=normalize_metadata(rec['dataset'],'dataset')
                ci=int(rec['incremental_id']);name=str(rec['class_name']).strip()
                if not name or name.lower()=='nan' or ds not in CLASS_ORDER or ci<0 or ci>=len(CLASS_ORDER[ds]):
                    raise ValueError(f'Invalid class label in {candidate}: {rec.to_dict()}')
                key=(ds,ci)
                if key in overrides and overrides[key]!=name:
                    raise ValueError(f'Conflicting labels for {key}')
                overrides[key]=name
    for ds in datasets:
        if ds not in CLASS_ORDER:continue
        dirs=['nwpu','nwpu_resisc45'] if ds=='nwpu' else ['aid']
        for dd in dirs:
            manifest=project_root/'splits'/dd/'seed_1993'/'train.csv'
            if not manifest.exists():continue
            # Source must explicitly contain the original class ID and class name.
            src=pd.read_csv(manifest,usecols=lambda col:col in {'original_class_id','class_name'})
            if not {'original_class_id','class_name'}.issubset(src.columns):continue
            by_original={}
            for original_id, grp in src.groupby('original_class_id'):
                names=set(grp.class_name.astype(str).str.strip())
                if len(names)!=1:continue
                by_original[int(original_id)]=names.pop()
            for inc,oid in enumerate(CLASS_ORDER[ds]):
                key=(ds,inc)
                if key not in overrides and oid in by_original:
                    overrides[key]=by_original[oid]
            break
    records=[]
    for ds in datasets:
        if ds not in CLASS_ORDER: continue
        for i,oid in enumerate(CLASS_ORDER[ds]):
            label=overrides.get((ds,i))
            records.append(dict(dataset=ds,incremental_id=i,original_class_id=oid,
                                class_name=label if label else f'C{i:02d}',
                                class_label_verified=bool(label),
                                label_source='provided_or_manifest' if label else 'incremental_id_only'))
    return pd.DataFrame(records)


def per_run_details(valid:pd.DataFrame,labels:pd.DataFrame):
    task_rows=[];class_rows=[];miss_rows=[];forget_rows=[]
    label_lookup={(r.dataset,int(r.incremental_id)):(r.class_name,bool(r.class_label_verified)) for r in labels.itertuples(index=False)}
    for run in valid.itertuples(index=False):
        meta={k:getattr(run,k) for k in METADATA_COLS}
        seen=run.seen_classes_curve
        histories=defaultdict(list)
        for t,n in enumerate(seen):
            cm=np.load(Path(run.run_dir)/'confusion_matrices'/f'task_{t:02d}.npy',allow_pickle=False)
            counts=cm.sum(axis=1)
            correct=np.diag(cm)
            joint=float(correct.sum()/counts.sum()*100) if counts.sum() else np.nan
            new_start=0 if t==0 else seen[t-1]
            n_old=int(counts[:new_start].sum())
            n_new=int(counts[new_start:].sum())
            task_rows.append(dict(**meta,task_id=t,seen_classes=n,top1_accuracy_pct=joint,
                                  old_accuracy_pct=float(correct[:new_start].sum()/n_old*100) if n_old else np.nan,
                                  new_accuracy_pct=float(correct[new_start:].sum()/n_new*100) if n_new else np.nan,
                                  n_test=int(counts.sum())))
            for cid in range(n):
                label,verified=label_lookup.get((run.dataset,cid),(f'C{cid:02d}',False))
                acc=float(correct[cid]/counts[cid]*100) if counts[cid] else np.nan
                introduced=next(i for i,s in enumerate(seen) if cid<s)
                cr=dict(**meta,task_id=t,class_id=cid,class_name=label,class_label_verified=verified,
                        introduced_task=introduced,is_new_class=t==introduced,
                        support=int(counts[cid]),correct=int(correct[cid]),class_accuracy_pct=acc)
                class_rows.append(cr)
                histories[cid].append((t,acc,int(counts[cid])))
            if t==len(seen)-1:
                for cid in range(n):
                    label,_=label_lookup.get((run.dataset,cid),(f'C{cid:02d}',False))
                    for pred_id in range(n):
                        if cid==pred_id or cm[cid,pred_id]==0:continue
                        pred_label,_=label_lookup.get((run.dataset,pred_id),(f'C{pred_id:02d}',False))
                        miss_rows.append(dict(**meta,true_class_id=cid,true_class=label,
                                              predicted_class_id=pred_id,predicted_class=pred_label,
                                              errors=int(cm[cid,pred_id]),true_support=int(counts[cid]),
                                              error_rate_pct=float(cm[cid,pred_id]/counts[cid]*100) if counts[cid] else np.nan))
        for cid,hist in histories.items():
            label,verified=label_lookup.get((run.dataset,cid),(f'C{cid:02d}',False))
            finite=[(t,a,s) for t,a,s in hist if np.isfinite(a)]
            if not finite:continue
            peak=max(a for _,a,_ in finite)
            final_acc=finite[-1][1]
            forget_rows.append(dict(**meta,class_id=cid,class_name=label,class_label_verified=verified,
                                   introduced_task=int(hist[0][0]),peak_accuracy_pct=peak,final_accuracy_pct=final_acc,
                                   class_forgetting_pp=peak-final_acc,final_support=finite[-1][2]))
    return (pd.DataFrame(task_rows),pd.DataFrame(class_rows),pd.DataFrame(miss_rows),pd.DataFrame(forget_rows))


def hist_equal(p1:Path,p2:Path,n_tasks):
    for t in range(n_tasks):
        a=np.load(p1/'confusion_matrices'/f'task_{t:02d}.npy',allow_pickle=False).sum(axis=1)
        b=np.load(p2/'confusion_matrices'/f'task_{t:02d}.npy',allow_pickle=False).sum(axis=1)
        if not np.array_equal(a,b):return False
    return True


def paired_metrics(valid:pd.DataFrame):
    rows=[];rejected=[]
    for key,grp in valid.groupby(KEYS,dropna=False):
        base=grp.loc[grp.attention == 'none']
        if len(base)!=1:continue
        b=base.iloc[0]
        for a in grp.loc[grp.attention !='none'].itertuples(index=False):
            reasons=[]
            if a.batch!=b.batch:reasons.append('batch_mismatch')
            if a.source_machine!=b.source_machine:reasons.append('machine_mismatch')
            if a.seen_classes_curve!=b.seen_classes_curve:reasons.append('class_progression_mismatch')
            if not hist_equal(Path(a.run_dir),Path(b.run_dir),int(b.n_tasks)):
                reasons.append('test_label_histogram_mismatch')
            base_info={k:getattr(a,k) for k in KEYS}
            base_info.update(attention=a.attention,source_machine=a.source_machine,batch=a.batch,
                             baseline_run_id=b.run_id,attention_run_id=a.run_id)
            if reasons:
                rejected.append({**base_info,'reasons':';'.join(reasons)})
                continue
            delta={}
            for name,col in [('aia','aia'),('lta','lta'),('macro_f1','macro_f1_pct'),('fr','fr'),('bwt','bwt')]:
                v=getattr(a,col); old=b[col]
                delta[f'delta_{name}_pp']=float(v-old) if v is not None and old is not None and np.isfinite(v) and np.isfinite(old) else np.nan
                delta[f'baseline_{name}_pct']=old
                delta[f'attention_{name}_pct']=v
            rows.append({**base_info,**delta})
    return pd.DataFrame(rows),pd.DataFrame(rejected)


def gain_summary(paired):
    if paired.empty:return pd.DataFrame()
    x=(paired.groupby(['dataset','method','attention'],as_index=False)
       .agg(n_pairs=('delta_aia_pp','size'),mean_delta_aia_pp=('delta_aia_pp','mean'),
            sd_delta_aia_pp=('delta_aia_pp','std'),mean_delta_lta_pp=('delta_lta_pp','mean'),
            mean_delta_macro_f1_pp=('delta_macro_f1_pp','mean'),
            mean_delta_fr_pp=('delta_fr_pp','mean'),
            wins=('delta_aia_pp',lambda s:int((s>0).sum()))))
    x['expected_pairs']=4
    x['complete_bundle']=x.n_pairs==x.expected_pairs
    x['win_rate_pct']=x.wins/x.n_pairs*100
    return x.sort_values(['dataset','mean_delta_aia_pp'],ascending=[True,False]).reset_index(drop=True)


def class_attention_comparisons(class_rows, valid):
    if class_rows.empty or valid.empty:return pd.DataFrame(),pd.DataFrame()
    keys=KEYS+['task_id','class_id']
    b=class_rows[class_rows.attention=='none'].copy()
    a=class_rows[class_rows.attention!='none'].copy()
    # Only fair paired experiments (checked before). Caller will filter via pair tuple.
    if b.empty or a.empty:return pd.DataFrame(),pd.DataFrame()
    pairs=a.merge(b[keys+['class_accuracy_pct','support','correct','run_id']],on=keys,
                  suffixes=('_attention','_none'),validate='many_to_one')
    pairs=pairs.rename(columns={'class_accuracy_pct_attention':'attention_class_accuracy_pct',
                                'class_accuracy_pct_none':'baseline_class_accuracy_pct',
                                'support_attention':'attention_support','support_none':'baseline_support',
                                'run_id_attention':'attention_run_id','run_id_none':'baseline_run_id'})
    pairs=pairs[pairs.attention_support==pairs.baseline_support].copy()
    pairs['delta_class_accuracy_pp']=pairs.attention_class_accuracy_pct-pairs.baseline_class_accuracy_pct
    wide=class_rows[class_rows.task_id==class_rows.groupby(METADATA_COLS,dropna=False).task_id.transform('max')].copy()
    dims=['dataset','method','protocol','backbone','seed','class_id','class_name']
    wide=wide.pivot_table(index=dims,columns='attention',values='class_accuracy_pct',aggfunc='first').reset_index()
    wide.columns.name=None
    support=class_rows[class_rows.task_id==class_rows.groupby(METADATA_COLS,dropna=False).task_id.transform('max')].groupby(dims,as_index=False).support.max()
    wide=wide.merge(support,on=dims,how='left',validate='one_to_one')
    for att in ATTENTIONS:
        if att not in wide:wide[att]=np.nan
        if att!='none':wide['delta_'+att+'_pp']=wide[att]-wide['none']
    return pairs,wide


def misclassification_changes(valid,labels,paired):
    if paired.empty:return pd.DataFrame()
    labels_by_ds={(r.dataset,int(r.incremental_id)):r.class_name for r in labels.itertuples(index=False)}
    lookup={(r.dataset,r.method,r.protocol,r.backbone,int(r.seed),r.attention):r for r in valid.itertuples(index=False)}
    records=[]
    for pair in paired.itertuples(index=False):
        key=(pair.dataset,pair.method,pair.protocol,pair.backbone,int(pair.seed))
        a=lookup[key+(pair.attention,)];b=lookup[key+('none',)]
        cm_a=np.load(Path(a.run_dir)/'final_confusion_matrix.npy',allow_pickle=False)
        cm_b=np.load(Path(b.run_dir)/'final_confusion_matrix.npy',allow_pickle=False)
        if not np.array_equal(cm_a.sum(axis=1),cm_b.sum(axis=1)):
            continue
        supports=cm_b.sum(axis=1)
        for i in range(cm_b.shape[0]):
            for j in range(cm_b.shape[1]):
                if i==j:continue
                old=int(cm_b[i,j]);new=int(cm_a[i,j])
                if not old and not new:continue
                support=int(supports[i]);
                records.append({**{k:getattr(pair,k) for k in KEYS},'attention':pair.attention,
                                'baseline_run_id':b.run_id,'attention_run_id':a.run_id,
                                'true_class_id':i,'true_class':labels_by_ds.get((pair.dataset,i),f'C{i:02d}'),
                                'predicted_class_id':j,'predicted_class':labels_by_ds.get((pair.dataset,j),f'C{j:02d}'),
                                'support':support,'errors_none':old,'errors_attention':new,
                                'errors_reduced':old-new,
                                'delta_error_rate_pp':(new-old)/support*100 if support else np.nan})
    return pd.DataFrame(records)

def task_and_forgetting_delta(tasks:pd.DataFrame,forget:pd.DataFrame,paired:pd.DataFrame):
    if paired.empty:return pd.DataFrame(),pd.DataFrame()
    tk=[];fk=[]
    for pair in paired.itertuples(index=False):
        where={k:getattr(pair,k) for k in KEYS}
        def fil(df,attention):
            if df.empty:return df
            sub=df.copy()
            for k,v in where.items():sub=sub[sub[k]==v]
            return sub[sub.attention==attention]
        a,b=fil(tasks,pair.attention),fil(tasks,'none')
        if not a.empty and not b.empty:
            z=a.merge(b,on=['dataset','method','protocol','backbone','seed','task_id'],suffixes=('_att','_none'),validate='one_to_one')
            for r in z.itertuples(index=False):
                x={**where,'attention':pair.attention,'task_id':int(r.task_id),
                   'baseline_run_id':pair.baseline_run_id,'attention_run_id':pair.attention_run_id}
                for metric in ('top1_accuracy_pct','old_accuracy_pct','new_accuracy_pct'):
                    left=getattr(r,metric+'_att');right=getattr(r,metric+'_none')
                    x['baseline_'+metric]=right;x['attention_'+metric]=left
                    x['delta_'+metric.replace('_accuracy_pct','')+'_pp']=(left-right) if pd.notna(left) and pd.notna(right) else np.nan
                tk.append(x)
        a,b=fil(forget,pair.attention),fil(forget,'none')
        if not a.empty and not b.empty:
            z=a.merge(b,on=['dataset','method','protocol','backbone','seed','class_id'],suffixes=('_att','_none'),validate='one_to_one')
            for r in z.itertuples(index=False):
                fk.append({**where,'attention':pair.attention,'class_id':int(r.class_id),
                           'class_name':r.class_name_att,
                           'baseline_run_id':pair.baseline_run_id,'attention_run_id':pair.attention_run_id,
                           'baseline_forgetting_pp':float(r.class_forgetting_pp_none),
                           'attention_forgetting_pp':float(r.class_forgetting_pp_att),
                           'delta_forgetting_pp':float(r.class_forgetting_pp_att-r.class_forgetting_pp_none),
                           'support':int(r.final_support_att)})
    return pd.DataFrame(tk),pd.DataFrame(fk)


def insights_from_evidence(coverage,valid,paired,gains,class_delta,forgetting,mis_delta,labels,
                           task_delta=None, forgetting_delta=None):
    """Rule-based observations, NOT LLM judgments or scientifically causal conclusions."""
    insights=[]
    def add(type_,title,claim,evidence,limitations,level='descriptive'):
        insights.append(dict(insight_id=f'I{len(insights)+1:03d}',type=type_,title=title,
                             claim=claim,evidence=evidence,limitations=limitations,
                             confidence_label=level))
    ready=int(coverage.ready.sum())
    total=len(coverage)
    add('coverage','Kelengkapan hasil production',
        f'{ready} dari {total} konfigurasi pada grid yang dipilih lolos audit artefak.',
        [{'file':'expected_coverage.csv','filter':'ready == True','ready':ready,'expected':total}],
        ['Hasil yang incomplete/running/failed tidak dipakai untuk menghitung metrik.',
         'ready_snapshot memakai inventory checkpoint dari mesin asal, bukan pemuatan weights.'],
        'final' if ready==total else 'preliminary')
    if not valid.empty:
        for protocol,group in valid.groupby('protocol'):
            champion=group.sort_values('aia',ascending=False).iloc[0]
            add('absolute_performance',f'AIA tertinggi yang teramati untuk {protocol}',
                f"{champion['method'].upper()} {champion['backbone'].upper()} {champion['attention'].upper()} pada {protocol} memiliki AIA {float(champion['aia']):.2f}% dari hasil valid yang tersedia.",
                [{'file':'valid_runs.csv','run_id':champion.run_id,'aia_pct':float(champion.aia),'lta_pct':float(champion.lta)}],
                ['Ini ranking AIA absolut, bukan efek attention.','Hanya berlaku untuk run yang sudah tersedia dan lolos audit.'],
                'final' if ready==total else 'preliminary')
    if not gains.empty:
        for ds,g in gains.groupby('dataset'):
            for a in ('eca','cbam','procbam'):
                subset=g[g.attention==a]
                if subset.empty:continue
                complete=subset[subset.complete_bundle]
                if complete.empty:
                    add('insufficient_pairs',f'{a.upper()} belum bisa diranking final pada {ds.upper()}',
                        'Belum ada metode dengan empat pasangan R18/R34 × T5/T10 yang lengkap untuk attention ini.',
                        [{'file':'method_attention_gain.csv','attention':a,'n_methods_observed':len(subset)}],
                        ['Rata-rata dari subset yang tersedia bisa bias; jangan menganggap n<4 sebagai ranking final.'],'preliminary')
                    continue
                best=complete.sort_values('mean_delta_aia_pp',ascending=False).iloc[0]
                add('attention_gain',f'Metode paling responsif terhadap {a.upper()} pada {ds.upper()} (bundle lengkap)',
                    f"{best['method'].upper()} mempunyai Mean ΔAIA {float(best.mean_delta_aia_pp):+.2f} pp dari {int(best.n_pairs)} pasangan sebanding.",
                    [{'file':'method_attention_gain.csv','dataset':ds,'method':best.method,
                      'attention':a,'n_pairs':int(best.n_pairs),'mean_delta_aia_pp':float(best.mean_delta_aia_pp)}],
                    ['Perbandingan yang belum memiliki empat pasangan pada metode lain tidak termasuk ranking final.',
                     'Tidak menyatakan signifikansi statistik karena eksperimen memakai satu seed.'],
                    'final' if len(complete)==len(METHODS) else 'preliminary')
        neg=paired.sort_values('delta_aia_pp').head(3)
        for r in neg.itertuples(index=False):
            if r.delta_aia_pp>=0:continue
            add('negative_attention_effect','Attention menurunkan AIA dalam satu kondisi',
                f'{r.method.upper()} {r.backbone.upper()} {r.protocol} + {r.attention.upper()} memiliki ΔAIA {r.delta_aia_pp:+.2f} pp vs None.',
                [{'file':'paired_attention_effects.csv','baseline_run_id':r.baseline_run_id,
                  'attention_run_id':r.attention_run_id,'delta_aia_pp':float(r.delta_aia_pp)}],
                ['Perbandingan deskriptif untuk satu kondisi, bukan kesimpulan umum.'])
    if not class_delta.empty:
        finals=class_delta.sort_values('task_id').groupby(['dataset','method','protocol','backbone','attention','seed','class_id'],as_index=False).tail(1)
        finals=finals[finals.attention_support>0]
        if not finals.empty:
            # Small supports can yield misleading 0/100% swings; require at least 10 examples for highlighted insights.
            finals=finals[finals.attention_support>=10]
            winners=finals[finals.delta_class_accuracy_pp>0].nlargest(5,'delta_class_accuracy_pp')
            losers=finals[finals.delta_class_accuracy_pp<0].nsmallest(5,'delta_class_accuracy_pp')
            for tag,sub in [('improvement',winners),('degradation',losers)]:
                for r in sub.itertuples(index=False):
                    add('per_class_'+tag,('Peningkatan' if tag=='improvement' else 'Penurunan')+' akurasi kelas di task final',
                        f'{r.method.upper()} {r.backbone.upper()} {r.protocol} / {r.attention.upper()}, kelas {r.class_name}: '+
                        f'{r.baseline_class_accuracy_pct:.2f}% menjadi {r.attention_class_accuracy_pct:.2f}% ({r.delta_class_accuracy_pp:+.2f} pp; n={int(r.attention_support)}).',
                        [{'file':'class_attention_delta.csv','baseline_run_id':r.baseline_run_id,
                          'attention_run_id':r.attention_run_id,'task_id':int(r.task_id),
                          'class_id':int(r.class_id),'support':int(r.attention_support),
                          'delta_class_accuracy_pp':float(r.delta_class_accuracy_pp)}],
                        ['Akurasi kelas di sini adalah recall kelas pada test set.',
                         'Hanya kelas dengan >=10 sampel test di-highlight.',
                         'Peningkatan/penurunan pada kelas tunggal tidak membuktikan fitur visual spesifik penyebabnya.'])
    if not mis_delta.empty:
        improved=mis_delta[mis_delta.errors_reduced>0].nlargest(5,'errors_reduced')
        for r in improved.itertuples(index=False):
            add('confusion_change','Kesalahan antar kelas berkurang',
                f'{r.method.upper()} {r.backbone.upper()} {r.protocol} + {r.attention.upper()}: '+
                f'kesalahan {r.true_class} → {r.predicted_class} berubah {r.errors_none} menjadi {r.errors_attention} '+
                f'(berkurang {int(r.errors_reduced)} dari n={int(r.support)} sampel kelas asal).',
                [{'file':'misclassification_delta.csv','baseline_run_id':r.baseline_run_id,'attention_run_id':r.attention_run_id,
                  'true_class_id':int(r.true_class_id),'predicted_class_id':int(r.predicted_class_id),
                  'errors_reduced':int(r.errors_reduced)}],
                ['Kelas yang tertukar belum tentu secara visual mirip; kesamaan visual perlu inspeksi citra.',
                 'Kesalahan ke kelas lain dapat meningkat; tinjau seluruh row confusion matrix.'])
    if task_delta is not None and not task_delta.empty:
        z=task_delta[task_delta.task_id>0].copy()
        if not z.empty:
            top=z.sort_values('delta_old_pp',ascending=False).dropna(subset=['delta_old_pp']).head(3)
            for r in top.itertuples(index=False):
                if r.delta_old_pp<=0:continue
                add('old_class_retention','Perubahan retensi kelas lama pada task tertentu',
                    f'{r.method.upper()} {r.backbone.upper()} {r.protocol} {r.attention.upper()} task {int(r.task_id)}: '
                    f'old-class accuracy naik {r.delta_old_pp:+.2f} pp vs None.',
                    [{'file':'task_attention_delta.csv','baseline_run_id':r.baseline_run_id,
                      'attention_run_id':r.attention_run_id,'task_id':int(r.task_id),
                      'delta_old_pp':float(r.delta_old_pp)}],
                    ['Retensi lebih baik pada satu task tidak otomatis membuktikan forgetting keseluruhan turun.'])
    if forgetting_delta is not None and not forgetting_delta.empty:
        z=forgetting_delta[(forgetting_delta.support>=10) & (forgetting_delta.delta_forgetting_pp < 0)]
        for r in z.nsmallest(3,'delta_forgetting_pp').itertuples(index=False):
            add('class_forgetting_improvement','Forgetting kelas tertentu berkurang',
                f'{r.method.upper()} {r.backbone.upper()} {r.protocol} {r.attention.upper()}: '
                f'{r.class_name} memiliki perubahan per-class forgetting {r.delta_forgetting_pp:+.2f} pp vs None.',
                [{'file':'class_forgetting_attention_delta.csv','baseline_run_id':r.baseline_run_id,
                  'attention_run_id':r.attention_run_id,'class_id':int(r.class_id),
                  'delta_forgetting_pp':float(r.delta_forgetting_pp)}],
                ['Forgetting diukur sejak task kelas pertama diperkenalkan.'])
    verified_names=int(labels.class_label_verified.sum()) if not labels.empty else 0
    label_count=len(labels)
    if verified_names<label_count:
        add('data_quality','Sebagian nama kelas belum dapat diverifikasi',
            f'{verified_names}/{label_count} pemetaan nama kelas tersedia; sisanya menggunakan kode incremental C00, C01, dst.',
            [{'file':'class_label_map.csv','verified_labels':verified_names,'total':label_count}],
            ['Jangan mengasumsikan kode C00 adalah nama kategori tertentu. Sediakan analysis_inputs/class_labels.csv bila diperlukan.'],
            'preliminary')
    return insights




# ============================================================================
# Exhaustive evidence (v2) — one row for every expected run and attention pair
# ============================================================================

def exhaustive_evidence(coverage, valid, paired, rejected, gains, tasks, class_delta,
                        task_delta, forgetting_delta, mis_delta):
    """Build exhaustive, traceable summaries; never choose only global top-k.

    Full AID scope: 80 model rows, 20 bundles, 60 attention-pair rows.
    Per-class/per-task detail stays in the original long-format CSVs.
    """
    key_fields = ['dataset','method','protocol','backbone','seed']
    run_fields = key_fields + ['attention']

    def identifier(r, fields):
        return tuple(getattr(r, f) if not isinstance(r, dict) else r[f] for f in fields)

    def as_number(value):
        v = numeric(value)
        return None if pd.isna(v) else float(v)

    def round_num(value, digits=4):
        v = as_number(value)
        return None if v is None else round(v, digits)

    def change_label(value, smaller_is_better=False):
        val = as_number(value)
        if val is None: return 'tidak tersedia'
        if abs(val) < 1e-9: return 'tidak berubah'
        good = val < 0 if smaller_is_better else val > 0
        return 'membaik' if good else 'memburuk'

    # Preserve every expected configuration even if it has not finished training.
    valid_index = {identifier(r, run_fields): r for r in valid.itertuples(index=False)} if not valid.empty else {}
    coverage_rows = []
    coverage_idx = {}
    for row in coverage.itertuples(index=False):
        key = identifier(row,run_fields)
        run = valid_index.get(key)
        item = {k:getattr(row,k) for k in run_fields}
        item.update({
            'status':str(row.coverage),
            'ready':bool(row.ready),
            'run_id':str(run.run_id) if run is not None else None,
            'source_machine':str(run.source_machine) if run is not None else None,
            'batch':int(run.batch) if run is not None else None,
            'aia_pct':round_num(run.aia) if run is not None else None,
            'lta_pct':round_num(run.lta) if run is not None else None,
            'macro_f1_pct':round_num(run.macro_f1_pct) if run is not None else None,
            'forgetting_rate_pct':round_num(run.fr) if run is not None else None,
            'bwt_pct':round_num(run.bwt) if run is not None else None,
            'evidence_file':'valid_runs.csv' if run is not None else 'expected_coverage.csv',
        })
        coverage_rows.append(item)
        coverage_idx[key]=item
    models_all=pd.DataFrame(coverage_rows)

    pair_idx={identifier(r,run_fields):r for r in paired.itertuples(index=False)} if not paired.empty else {}
    rejected_idx={identifier(r,run_fields):r for r in rejected.itertuples(index=False)} if not rejected.empty else {}
    pair_rows=[]
    # Per-class final observations are exhaustive, not merely top-5 samples.
    class_final=class_delta.copy()
    if not class_final.empty:
        class_final=class_final[class_final.task_id == class_final.protocol.map({'T5':4,'T10':9})].copy()
        class_final=class_final.sort_values(run_fields+['class_id']).reset_index(drop=True)

    for key in sorted(coverage_idx):
        dataset,method,protocol,backbone,seed,attention=key
        if attention=='none':continue
        baseline_key=(dataset,method,protocol,backbone,seed,'none')
        att=coverage_idx[key]
        base=coverage_idx[baseline_key]
        found=pair_idx.get(key)
        rejected_item=rejected_idx.get(key)
        status='paired_valid' if found is not None else (
            'rejected_fairness' if rejected_item is not None else
            'baseline_unavailable' if not base['ready'] else
            'attention_unavailable' if not att['ready'] else 'unpaired_review_required')
        rec={k:v for k,v in zip(run_fields,key)}
        rec.update({
            'status':status,
            'baseline_status':base['status'],
            'attention_status':att['status'],
            'baseline_run_id':base['run_id'],
            'attention_run_id':att['run_id'],
            'source_machine':att['source_machine'],
            'batch':att['batch'],
            'baseline_aia_pct':base['aia_pct'],
            'attention_aia_pct':att['aia_pct'],
            'baseline_lta_pct':base['lta_pct'],
            'attention_lta_pct':att['lta_pct'],
            'delta_aia_pp':round_num(found.delta_aia_pp) if found is not None else None,
            'delta_lta_pp':round_num(found.delta_lta_pp) if found is not None else None,
            'delta_macro_f1_pp':round_num(found.delta_macro_f1_pp) if found is not None else None,
            'delta_fr_pp':round_num(found.delta_fr_pp) if found is not None else None,
            'delta_bwt_pp':round_num(found.delta_bwt_pp) if found is not None else None,
            'aia_interpretation':change_label(found.delta_aia_pp) if found is not None else 'tidak dapat dibandingkan',
            'lta_interpretation':change_label(found.delta_lta_pp) if found is not None else 'tidak dapat dibandingkan',
            'fairness_issue':str(rejected_item.reasons) if rejected_item is not None else None,
            'evidence_file':'paired_attention_effects.csv' if found is not None else (
                'rejected_pairs.csv' if rejected_item is not None else 'expected_coverage.csv'),
        })
        if found is not None:
            def subset(df):
                if df.empty:return df
                mask=pd.Series(True,index=df.index)
                for f,v in zip(run_fields,key):mask &= (df[f]==v)
                return df[mask]
            cls=subset(class_final)
            rec['class_count_analyzed']=int(len(cls))
            rec['classes_improved']=int((cls.delta_class_accuracy_pp>1e-9).sum()) if not cls.empty else 0
            rec['classes_degraded']=int((cls.delta_class_accuracy_pp < -1e-9).sum()) if not cls.empty else 0
            rec['classes_unchanged']=int((cls.delta_class_accuracy_pp.abs()<=1e-9).sum()) if not cls.empty else 0
            rec['mean_delta_class_accuracy_pp']=round_num(cls.delta_class_accuracy_pp.mean()) if not cls.empty else None
            if not cls.empty:
                best=cls.sort_values(['delta_class_accuracy_pp','class_id'],ascending=[False,True]).iloc[0]
                worst=cls.sort_values(['delta_class_accuracy_pp','class_id'],ascending=[True,True]).iloc[0]
                rec.update(best_class_name=str(best.class_name),best_class_id=int(best.class_id),
                           best_class_delta_pp=round_num(best.delta_class_accuracy_pp),
                           worst_class_name=str(worst.class_name),worst_class_id=int(worst.class_id),
                           worst_class_delta_pp=round_num(worst.delta_class_accuracy_pp))
            t=subset(task_delta)
            rec['tasks_compared']=int(len(t))
            t_new=t[t.task_id>0] if not t.empty else t
            rec['mean_delta_old_pp']=round_num(t_new.delta_old_pp.mean()) if not t_new.empty else None
            rec['mean_delta_new_pp']=round_num(t_new.delta_new_pp.mean()) if not t_new.empty else None
            f=subset(forgetting_delta)
            rec['forgetting_classes_compared']=int(len(f))
            if not f.empty:
                rec['classes_less_forgetting']=int((f.delta_forgetting_pp < -1e-9).sum())
                rec['classes_more_forgetting']=int((f.delta_forgetting_pp > 1e-9).sum())
                rec['classes_same_forgetting']=int((f.delta_forgetting_pp.abs()<=1e-9).sum())
                rec['mean_delta_class_forgetting_pp']=round_num(f.delta_forgetting_pp.mean())
            mis=subset(mis_delta)
            rec['confusion_directions_changed']=int((mis.errors_reduced!=0).sum()) if not mis.empty else 0
            if not mis.empty:
                reduced=mis[mis.errors_reduced>0]
                increased=mis[mis.errors_reduced<0]
                rec['directions_errors_reduced']=int(len(reduced))
                rec['directions_errors_increased']=int(len(increased))
                if not reduced.empty:
                    best_m=reduced.sort_values('errors_reduced',ascending=False).iloc[0]
                    rec['max_reduced_confusion']=str(best_m.true_class)+' -> '+str(best_m.predicted_class)
                    rec['max_reduced_errors']=int(best_m.errors_reduced)
                if not increased.empty:
                    worst_m=increased.sort_values('errors_reduced').iloc[0]
                    rec['max_increased_confusion']=str(worst_m.true_class)+' -> '+str(worst_m.predicted_class)
                    rec['max_increased_errors']=int(-worst_m.errors_reduced)
        pair_rows.append(rec)
    pairs_all=pd.DataFrame(pair_rows)

    bundles=[]
    for (ds,method,protocol,backbone,seed),grp in models_all.groupby(key_fields,sort=True):
        row={k:v for k,v in zip(key_fields,(ds,method,protocol,backbone,seed))}
        row.update(n_ready=int(grp.ready.sum()),n_expected=4,
                   complete=bool(grp.ready.all()),
                   n_fair_pairs=int((pairs_all[(pairs_all.dataset==ds)&(pairs_all.method==method)&
                                         (pairs_all.protocol==protocol)&(pairs_all.backbone==backbone)&
                                         (pairs_all.seed==seed)].status=='paired_valid').sum()))
        for attention in ATTENTIONS:
            item=grp[grp.attention==attention].iloc[0]
            row[f'aia_{attention}_pct']=item.aia_pct
            row[f'lta_{attention}_pct']=item.lta_pct
            row[f'run_{attention}_id']=item.run_id
        bundles.append(row)
    bundles_all=pd.DataFrame(bundles)
    expected_runs=len(coverage)
    expected_pairs=expected_runs//4*3
    if len(models_all)!=expected_runs or len(pairs_all)!=expected_pairs or len(bundles_all)!=expected_runs//4:
        raise RuntimeError('Exhaustive coverage mismatch: expected all model, bundle and pair rows')
    if bool(models_all.ready.all()):
        if (pairs_all.status!='paired_valid').any():
            raise RuntimeError('All model runs ready but some attention pairs are not valid; inspect fairness')
        if (pairs_all.class_count_analyzed != pairs_all.dataset.map({'aid':30,'nwpu':45})).any():
            raise RuntimeError('All pairs valid but final per-class coverage is incomplete')
        if (pairs_all.tasks_compared != pairs_all.protocol.map({'T5':5,'T10':10})).any():
            raise RuntimeError('All pairs valid but per-task delta coverage is incomplete')
    return models_all,bundles_all,pairs_all,class_final


def exhaustive_markdown(models,bundles,pairs,class_final):
    """Markdown covers every run and every baseline-attention comparison."""
    def fmt(value,unit='%',n=2):
        v=numeric(value)
        return 'N/A' if pd.isna(v) else f'{float(v):+.{n}f}{unit}' if unit==' pp' else f'{float(v):.{n}f}{unit}'
    def safe(value):
        return 'N/A' if value is None or (isinstance(value,float) and pd.isna(value)) else str(value)
    out=[
        '## ANALISIS LENGKAP — SEMUA MODEL DAN SEMUA PERBANDINGAN',
        '',
        'Bagian ini **tidak memakai top-k selection**: setiap konfigurasi dan setiap pasangan attention vs None memiliki baris tersendiri. '
        'Insight terperinci seluruh kelas dan seluruh task ada pada CSV long-format. ',
        f'Jumlah model tercatat: **{len(models)}**; bundle: **{len(bundles)}**; pasangan terdaftar: **{len(pairs)}**; '
        f'pasangan yang benar-benar dapat dibandingkan: **{int((pairs.status=="paired_valid").sum())}**.',
        '',
        '### Seluruh konfigurasi model (termasuk baseline None)',
        '',
        '| Dataset | Metode | T | Backbone | Attention | Status | AIA (%) | LTA (%) | Macro-F1 (%) | FR | BWT | Run ID |',
        '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|',
    ]
    for r in models.itertuples(index=False):
        out.append(f'| {r.dataset} | {r.method.upper()} | {r.protocol} | {r.backbone.upper()} | {r.attention.upper()} | '
                   f'{r.status} | {fmt(r.aia_pct)} | {fmt(r.lta_pct)} | {fmt(r.macro_f1_pct)} | '
                   f'{fmt(r.forgetting_rate_pct)} | {fmt(r.bwt_pct)} | `{safe(r.run_id)}` |')
    out+=['', '### Seluruh bundle dan empat attention', '',
          '| Dataset | Metode | T | Backbone | Lengkap | Paired | AIA None | AIA ECA | AIA CBAM | AIA ProCBAM |',
          '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in bundles.itertuples(index=False):
        out.append(f'| {r.dataset} | {r.method.upper()} | {r.protocol} | {r.backbone.upper()} | '
                   f'{int(r.n_ready)}/4 | {int(r.n_fair_pairs)}/3 | '
                   f'{fmt(r.aia_none_pct)} | {fmt(r.aia_eca_pct)} | {fmt(r.aia_cbam_pct)} | {fmt(r.aia_procbam_pct)} |')
    out+=['', '### Perbandingan individual untuk SEMUA attention', '',
          'Angka delta dalam **percentage points (pp)**. FR negatif berarti *forgetting rate* numerik lebih rendah; '
          'BWT perlu dibaca menurut konvensi runner. Klaim tentang retensi harus didukung metrik lain, bukan satu angka saja.', '']
    for (ds,method,protocol,backbone,seed), g in pairs.groupby(['dataset','method','protocol','backbone','seed'],sort=True):
        out.append(f'#### {ds.upper()} — {method.upper()} {backbone.upper()} {protocol}, seed {seed}')
        out.append('')
        out.append('| Attention | Status | AIA None → Attention | ΔAIA | LTA None → Attention | ΔLTA | ΔMacro-F1 | ΔFR | ΔBWT |')
        out.append('|---|---|---|---:|---|---:|---:|---:|---:|')
        for r in g.itertuples(index=False):
            out.append(f'| {r.attention.upper()} | {r.status} | {fmt(r.baseline_aia_pct)} → {fmt(r.attention_aia_pct)} | '
                       f'{fmt(r.delta_aia_pp," pp")} | {fmt(r.baseline_lta_pct)} → {fmt(r.attention_lta_pct)} | '
                       f'{fmt(r.delta_lta_pp," pp")} | {fmt(r.delta_macro_f1_pp," pp")} | '
                       f'{fmt(r.delta_fr_pp," pp")} | {fmt(r.delta_bwt_pp," pp")} |')
        out.append('')
        for r in g.itertuples(index=False):
            if r.status!='paired_valid':
                out.append(f'- **{r.attention.upper()} — tidak tersedia** ({r.status}; baseline={r.baseline_status}, perhatian={r.attention_status}, '
                           f'fairness={safe(r.fairness_issue)}). Jangan menyimpulkan efeknya.')
                continue
            out.append(
                f'- **{r.attention.upper()}** (`{safe(r.baseline_run_id)}` → `{safe(r.attention_run_id)}`): '
                f'ΔAIA {fmt(r.delta_aia_pp," pp")} ({r.aia_interpretation}), '
                f'ΔLTA {fmt(r.delta_lta_pp," pp")} ({r.lta_interpretation}); '
                f'class accuracy task akhir: **{int(r.classes_improved)} naik, {int(r.classes_degraded)} turun, '
                f'{int(r.classes_unchanged)} tetap** dari {int(r.class_count_analyzed)} kelas. '
                f'Kenaikan kelas terbesar: {safe(getattr(r,"best_class_name",None))} '
                f'({fmt(getattr(r,"best_class_delta_pp",None)," pp")}); '
                f'penurunan terbesar: {safe(getattr(r,"worst_class_name",None))} '
                f'({fmt(getattr(r,"worst_class_delta_pp",None)," pp")}). '
                f'Mean Δold/Δnew task 1+: {fmt(getattr(r,"mean_delta_old_pp",None)," pp")} / '
                f'{fmt(getattr(r,"mean_delta_new_pp",None)," pp")}; '
                f'forgetting kelas: {int(getattr(r,"classes_less_forgetting",0))} membaik, '
                f'{int(getattr(r,"classes_more_forgetting",0))} memburuk. '
                f'Arah confusion paling berkurang: {safe(getattr(r,"max_reduced_confusion",None))} '
                f'(n={safe(getattr(r,"max_reduced_errors",None))}); '
                f'paling bertambah: {safe(getattr(r,"max_increased_confusion",None))} '
                f'(n={safe(getattr(r,"max_increased_errors",None))}). '
                'Rujukan: `paired_attention_effects.csv`, `class_attention_delta.csv`, '
                '`task_attention_delta.csv`, `class_forgetting_attention_delta.csv`, `misclassification_delta.csv`.'
            )
        out.append('')
    out += [
        '### Seluruh class accuracy pada task akhir',
        '',
        f'Terdapat **{len(class_final)}** rekaman per-class final untuk semua pasangan yang valid. '
        'Lihat `class_insights_final_all.csv` dan `class_attention_delta.csv` untuk angka tiap kelas tanpa pemotongan baris; '
        'lihat `class_accuracy_long.csv` untuk setiap task. '
        'Untuk perbandingan confusion lengkap, lihat `misclassification_delta.csv` (seluruh arah dengan nonzero error).',
        '',
        '### Keterbatasan',
        '',
        'Angka adalah deskriptif pada seed utama 1993; 4 pasangan lintas backbone/protocol bukan 4 seed. '
        'Perubahan akurasi per kelas bukan penjelasan kausal atas fitur visual. '
        'Walaupun histogram label identik, dataset dan split manifest tetap perlu dipastikan konsisten secara terpisah.',
        '',
    ]
    return '\n'.join(out)+'\n'


def report_markdown(manifest,insights):
    n=manifest['counts']
    lines=[
        '# AI-Ready Evidence Report — EFCIL × Attention',
        '',f"Tanggal ekspor: {manifest['generated_at']}",
        '',f"Scope dataset: **{', '.join(manifest['dataset_scope'])}**.",
        f"Hasil valid: **{n['valid_runs']}**; total grid: **{n['expected_runs']}**; ready: **{n['ready_expected']}**.",
        f"Paired attention comparisons: **{n['paired_effects']}**; rejected fair pairs: **{n['rejected_pairs']}**.",
        '', '## Highlight otomatis (pilihan contoh, bukan seluruh model)',
        '', 'Temuan di bawah adalah observasi komputasional, **bukan interpretasi ilmiah otomatis atau klaim sebab-akibat**.', ''
    ]
    for item in insights:
        lines.append(f"### {item['insight_id']} — {item['title']}")
        lines.append(item['claim'])
        lines.append('Bukti: '+', '.join(e['file'] for e in item['evidence']))
        lines.append('Batasan: '+'; '.join(item['limitations']))
        lines.append('')
    lines += [
        '## Aturan interpretasi penting',
        '',
        '- `class_accuracy_pct` merupakan recall suatu kelas = TP / jumlah label benar kelas tersebut × 100.',
        '- Semua delta akurasi memakai satuan **percentage points (pp)**, bukan perubahan relatif persen.',
        '- Pembandingan perhatian yang sah harus sama dataset, method, protocol, backbone, seed, machine, batch, dan histogram label uji per task.',
        '- `n_pairs=4` pada AID berarti dua backbone × dua protocol, bukan empat seed independen.',
        '- Mean ΔAIA tidak mengindikasikan signifikansi statistik; seed utama tunggal 1993.',
        '- `FWT` tidak dipaksakan pada protokol dynamic classifier skripsi ini.',
        '- Rekaman snapshot checkpoints hanya inventory dari mesin asal, tidak membuktikan weights dapat dimuat.',
        '- Temuan bukan bukti bahwa attention menyebabkan penurunan catastrophic forgetting ataupun memperbaiki high inter-class similarity tanpa analisis lanjutan.',
        '- Class label yang tidak tersedia diverifikasi tetap ditulis Cxx, bukan nama scene hasil tebakan.',
        '', '## Cara meminta AI menganalisis hasil',
        '', 'Unggah **ai_ready_evaluation_bundle.zip**, lalu gunakan prompt pada `AI_PROMPT.md`.',
        'AI dapat memperluas pembahasan Bab IV dengan menghitung ulang/menyilang tabel, tetapi harus menyebut `run_id`, file bukti, jumlah pasangan dan batasan.',
    ]
    return '\n'.join(lines)+'\n'


PROMPT_TEXT="""# Prompt untuk AI — analisis Bab IV skripsi

Saya sedang meneliti integrasi ECA, CBAM, dan ProCBAM pada lima metode EFCIL
(FeTrIL, FeCAM, SSRE, EFC, PASS), dengan ResNet-18/34 dan protocol T5/T10,
untuk scene classification citra satelit. File yang saya upload adalah keluaran
notebook `08_ai_ready_insights.ipynb` dari hasil production terverifikasi.

Tolong:
1. Baca `manifest.json`, `data_dictionary.json`, `insights.json`, `run_audit.csv`,
   `expected_coverage.csv`, dan semua CSV yang relevan. Sebutkan kelengkapan data.
2. Jawab apakah setiap attention meningkatkan AIA dan LTA dibanding None pada
   pasangan method/dataset/protocol/backbone/seed/batch/machine yang sah.
3. WAJIB bahas seluruh 80 konfigurasi dan seluruh 60 pasangan AID, bukan hanya highlight.
   Baca model_insights_all.csv, attention_insights_all.csv, bundle_insights_all.csv,
   class_insights_final_all.csv, dan insights_all.json. Jika hasil partial, nyatakan yang missing.
   Untuk setiap attention, tentukan metode EFCIL paling responsif menggunakan
   Mean ΔAIA. Pisahkan hasil final (n_pairs=4 per method per dataset) dan preliminary.
4. Analisis class accuracy task akhir dan across tasks, per-class forgetting,
   old vs new accuracy, Macro-F1, dan pasangan misclassification yang berubah.
5. Susun temuan yang benar-benar didukung data lengkap dengan angka, `run_id`,
   nama CSV dan kolom yang mendukungnya. Jelaskan juga fenomena penurunan kinerja.
6. Buat outline hasil dan pembahasan Bab IV, membedakan hasil empiris dari hipotesis
   tentang kemiripan visual kelas. Jangan mengarang nama kelas yang masih Cxx.

Jangan memperlakukan smoke test sebagai hasil final, jangan mengklaim sebab-akibat
atau signifikansi statistik tanpa pengujian yang mendukung. Semua delta akurasi
berunit percentage points. Mean ΔAIA dari empat kondisi bukan empat seed independen.
Jika menemukan inkonsistensi data, laporkan alih-alih memilih angka secara sepihak.
"""


def main(project_root:Path, sources:dict[str,Path]|None=None, output_dir:Path|None=None,
         dataset_scope:str='aid', seed:int=1993, class_labels_file:Path|None=None,
         create_zip:bool=True, allow_inventory:bool=True):
    global ALLOW_CHECKPOINT_INVENTORY
    ALLOW_CHECKPOINT_INVENTORY=bool(allow_inventory)
    project_root=Path(project_root).resolve()
    if sources is None:
        sources={'pc2':project_root/'outputs'/'experiments',
                 'server':project_root/'analysis_inputs'/'server'/'experiments'}
    sources={k:Path(v).resolve() for k,v in sources.items()}
    output_dir=Path(output_dir).resolve() if output_dir is not None else project_root/'outputs'/'analysis_ai_ready'
    output_dir.mkdir(parents=True,exist_ok=True)
    ds=['aid','nwpu'] if dataset_scope in (None,'all') else [dataset_scope]
    for dataset in ds:
        if dataset not in CLASS_ORDER:raise ValueError('Unsupported dataset '+str(dataset))
    runs,valid=scan_sources(sources,dataset_scope)
    if not valid.empty:
        valid=valid[valid.seed==seed].copy()  # preserve other seeds in audit, not pooled.
    coverage=expected_coverage(runs,valid,dataset_scope,seed)
    labels=load_class_labels(project_root,ds,class_labels_file)
    tasks,classes,mis,forget=per_run_details(valid,labels) if not valid.empty else (pd.DataFrame(),)*4
    paired,rejected=paired_metrics(valid) if not valid.empty else (pd.DataFrame(),pd.DataFrame())
    gain=gain_summary(paired)
    class_delta,class_wide=class_attention_comparisons(classes,valid)
    # IMPORTANT: Do not compare classes from rejected/unfair pairs.
    if not class_delta.empty:
        fair={(r.dataset,r.method,r.protocol,r.backbone,int(r.seed),r.attention,r.baseline_run_id,r.attention_run_id)
              for r in paired.itertuples(index=False)}
        keep=class_delta.apply(lambda r:(r['dataset'],r['method'],r['protocol'],r['backbone'],int(r['seed']),
                                         r['attention'],r['baseline_run_id'],r['attention_run_id']) in fair,axis=1)
        class_delta=class_delta[keep].copy()
        # avoid reporting deltas in wide for unfair pairs
        fair_wide_cols={}
        for a in ('eca','cbam','procbam'):
            if a in class_wide:
                fair_keys={(r.dataset,r.method,r.protocol,r.backbone,int(r.seed)) for r in paired[paired.attention==a].itertuples(index=False)}
                class_wide['delta_'+a+'_pp']=[
                    row['delta_'+a+'_pp'] if (row['dataset'],row['method'],row['protocol'],row['backbone'],int(row['seed'])) in fair_keys else np.nan
                    for _,row in class_wide.iterrows()]
    miss_delta=misclassification_changes(valid,labels,paired)
    task_delta,forgetting_delta=task_and_forgetting_delta(tasks,forget,paired)
    model_insights,bundle_insights,pair_insights,all_class_final=exhaustive_evidence(
        coverage,valid,paired,rejected,gain,tasks,class_delta,task_delta,forgetting_delta,miss_delta)
    # Never serialize absolute file-system paths into the AI-shared report.
    drop_cols={'run_dir','task_curve','seen_classes_curve'}
    audited=runs.drop(columns=[c for c in drop_cols if c in runs],errors='ignore') if not runs.empty else pd.DataFrame()
    safe_valid=valid.drop(columns=[c for c in drop_cols if c in valid],errors='ignore') if not valid.empty else pd.DataFrame()
    tables={
        'run_audit.csv':(audited,['source_machine','run_id','status','issue','eligible']),
        'expected_coverage.csv':(coverage,['dataset','method','protocol','backbone','attention','coverage','ready']),
        'valid_runs.csv':(safe_valid,METADATA_COLS+['aia','lta','macro_f1_pct','fr','bwt']),
        'paired_attention_effects.csv':(paired,KEYS+['attention','delta_aia_pp','delta_lta_pp']),
        'rejected_pairs.csv':(rejected,KEYS+['attention','reasons']),
        'method_attention_gain.csv':(gain,['dataset','method','attention','n_pairs','mean_delta_aia_pp','complete_bundle']),
        'task_accuracy_long.csv':(tasks,METADATA_COLS+['task_id','top1_accuracy_pct','old_accuracy_pct','new_accuracy_pct']),
        'class_accuracy_long.csv':(classes,METADATA_COLS+['task_id','class_id','class_name','support','class_accuracy_pct']),
        'class_forgetting.csv':(forget,METADATA_COLS+['class_id','class_forgetting_pp']),
        'class_attention_delta.csv':(class_delta,KEYS+['attention','task_id','class_id','delta_class_accuracy_pp']),
        'class_attention_final_wide.csv':(class_wide,KEYS+['class_id','class_name','none','eca','cbam','procbam']),
        'misclassification_long.csv':(mis,METADATA_COLS+['true_class_id','predicted_class_id','errors']),
        'misclassification_delta.csv':(miss_delta,KEYS+['attention','true_class_id','predicted_class_id','errors_reduced']),
        'class_label_map.csv':(labels,['dataset','incremental_id','original_class_id','class_name','class_label_verified']),
        'task_attention_delta.csv':(task_delta,KEYS+['attention','task_id','delta_old_pp','delta_new_pp']),
        'class_forgetting_attention_delta.csv':(forgetting_delta,KEYS+['attention','class_id','delta_forgetting_pp']),
        'model_insights_all.csv':(model_insights,KEYS+['attention','status','aia_pct','lta_pct']),
        'bundle_insights_all.csv':(bundle_insights,KEYS+['n_ready','complete']),
        'attention_insights_all.csv':(pair_insights,KEYS+['attention','status','delta_aia_pp','delta_lta_pp']),
        'class_insights_final_all.csv':(all_class_final,KEYS+['attention','class_id','class_name','delta_class_accuracy_pp'])
    }
    for name,(data,cols) in tables.items():csv_write(data,output_dir/name,cols)
    manifest={
        'schema_version':AI_READY_EXPORTER_VERSION,
        'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'dataset_scope':ds,'seed':seed,
        'counts':{'candidate_runs':int(len(runs)),'valid_runs':int(len(valid)),
                  'expected_runs':int(len(coverage)),'ready_expected':int(coverage.ready.sum()),
                  'paired_effects':int(len(paired)),'rejected_pairs':int(len(rejected)),
                  'class_observations':int(len(classes)),'misclassification_pairs':int(len(miss_delta))},
        'sources':{machine:{'present':path.is_dir()} for machine,path in sources.items()},
        'ready_full_count':int((valid.status=='ready_full').sum()) if not valid.empty else 0,
        'ready_snapshot_count':int((valid.status=='ready_snapshot').sum()) if not valid.empty else 0,
        'checkpoint_policy':'snapshot inventory is not proof of torch.load success',
        'units':{'accuracy':'percent','delta_accuracy':'percentage points','counts':'samples'},
        'cautions':['One training seed 1993; no inference of statistical significance.',
                    'FWT N/A for dynamic classifier protocol.',
                    'Mean ΔAIA over four conditions is not across four independent seeds.',
                    'Dataset scope may only include AID.'],
    }
    insights=insights_from_evidence(coverage,valid,paired,gain,class_delta,forget,miss_delta,labels,
                                    task_delta,forgetting_delta)
    manifest['counts'].update({
        'model_insights_all':len(model_insights),
        'bundle_insights_all':len(bundle_insights),
        'expected_attention_pairs':len(pair_insights),
        'paired_insights_valid':int((pair_insights.status=='paired_valid').sum()),
        'class_insights_final_all':len(all_class_final),
    })
    comprehensive=exhaustive_markdown(model_insights,bundle_insights,pair_insights,all_class_final)
    jwrite(output_dir/'insights_all.json',{
        'schema_version':AI_READY_EXPORTER_VERSION,
        'description':'Every run and every expected attention comparison, including missing conditions; no global top-k cutoff',
        'counts':manifest['counts'],
        'run_insights':model_insights.to_dict('records'),
        'bundle_insights':bundle_insights.to_dict('records'),
        'attention_pair_insights':pair_insights.to_dict('records'),
        'class_level_full_table':'class_insights_final_all.csv',
        'task_level_full_table':'task_attention_delta.csv',
    })
    jwrite(output_dir/'manifest.json',manifest)
    jwrite(output_dir/'insights.json',{'generation':'deterministic_rules_no_llm',
                                      'insights':insights,'count':len(insights)})
    jwrite(output_dir/'data_dictionary.json',{
        'files':{**SOURCE_TABLES,
                 'model_insights_all.csv':'Satu baris untuk setiap expected model termasuk None dan yang missing.',
                 'bundle_insights_all.csv':'Semua 20 bundle AID method x backbone x T.',
                 'attention_insights_all.csv':'Satu baris per setiap expected attention-vs-none pair, dengan ringkasan seluruh dimensi metrik.',
                 'class_insights_final_all.csv':'Semua perubahan class recall pada task final, tanpa memilih top-k.',
                 'insights_all.json':'Ringkasan JSON per model, per bundle, dan per pair secara penuh.'},
        'column_notes':{
            'run_id':'Identitas model production untuk tracing source.',
            'eligible':'Hanya run berstatus ready_full/ready_snapshot dengan metrik tervalidasi.',
            'class_id':'Incremental class ID, bukan original class ID.',
            'class_name':'Nama kelas tervalidasi jika tersedia; Cxx jika belum.',
            'support':'Jumlah test sample untuk true class.',
            'class_accuracy_pct':'True positives / actual class support * 100; class recall.',
            'delta_class_accuracy_pp':'Class accuracy attention minus none (percentage points).',
            'delta_aia_pp':'AIA attention minus baseline None, in percentage points.',
            'errors_reduced':'Misclassification count from None minus from attention; positive is fewer errors.',
            'delta_error_rate_pp':'Row normalized error rate attention minus none; negative is fewer errors.',
            'complete_bundle':'Four matched pairs (R18+R34 x T5+T10) for one method/dataset/attention.',
            'n_pairs':'Number of fair matched baseline-attention comparisons, not independent training seeds.',
            'fr':'Canonical forgetting rate from runner; interpretation follows project definition.',
            'fwt':'Not applicable under dynamic classifier protocol.'},
        'data_sources':['Canonical per-task confusion matrix and y_true/y_pred archive',
                        'continual_metrics.json','effective_args.json','provenance.json'],
        'eligibility_rule':'Only completed experiment with all task artifacts and checkpoints or source checkpoint inventory; audit verifies confusion against saved predictions.'})
    base_report=report_markdown(manifest,insights)
    base_report=base_report.replace('## Highlight otomatis (pilihan contoh, bukan seluruh model)',comprehensive+'\n## Highlight otomatis (pilihan contoh, bukan seluruh model)',1)
    (output_dir/'AI_EVIDENCE_REPORT.md').write_text(base_report,encoding='utf-8')
    (output_dir/'AI_PROMPT.md').write_text(PROMPT_TEXT,encoding='utf-8')
    (output_dir/'README.md').write_text(
        '# AI-ready bundle\n\nCSV/JSON hasil production skripsi yang sudah diaudit dan siap diupload ke AI.\n\n'
        'Mulai dari `manifest.json`, `AI_EVIDENCE_REPORT.md`, `insights_all.json`, `AI_PROMPT.md`, `data_dictionary.json`, serta seluruh tabel CSV.\n\n'
        'Tidak termasuk dataset mentah, predictions individual, checkpoint weights, atau gambar satelit.\n'
        'Highlight di `insights.json` hanya contoh. Analisis SEMUA 80 model/60 pasangan AID ada di `insights_all.json`, `model_insights_all.csv` dan `attention_insights_all.csv`.\n'
        'Per-class penuh tersimpan pada `class_insights_final_all.csv` dan `class_attention_delta.csv`.\n'
        'Insight otomatis deterministik, bukan oleh LLM. Analisis interpretasi tetap memerlukan AI dan verifikasi manusia.\n',encoding='utf-8')
    zip_path=None
    if create_zip:
        zip_path=output_dir/'ai_ready_evaluation_bundle.zip'
        tmp=output_dir/'ai_ready_evaluation_bundle.zip.partial'
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True,compresslevel=6) as z:
            for fn in [*tables,'manifest.json','insights.json','data_dictionary.json','AI_EVIDENCE_REPORT.md','insights_all.json','AI_PROMPT.md','README.md']:
                z.write(output_dir/fn,arcname=fn)
        tmp.replace(zip_path)
        with zipfile.ZipFile(zip_path) as z:
            bad=z.testzip()
            if bad:raise ValueError(f'ZIP CRC error: {bad}')
    # Fail at generation time rather than leaving the notebook to fail on pd.read_csv.
    missing_required = [name for name in REQUIRED_ALL_MODELS_OUTPUTS
                        if not (output_dir / name).is_file()
                        or (output_dir / name).stat().st_size == 0]
    if missing_required:
        raise RuntimeError(
            "Exporter " + AI_READY_EXPORTER_VERSION
            + " gagal menghasilkan output ALL MODELS: "
            + ", ".join(missing_required)
            + ". Periksa error di cell/exporter sebelumnya."
        )
    return {'output_dir':str(output_dir),'zip_path':str(zip_path) if zip_path else None,
            'manifest':manifest,'insights_count':len(insights),'exhaustive_pairs':len(pair_insights),'files':list(tables)}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Export AI-readable, audited EFCIL evaluation reports')
    ap.add_argument('--project-root',default='.')
    ap.add_argument('--dataset-scope',choices=('aid','nwpu','all'),default='aid')
    ap.add_argument('--seed',type=int,default=1993)
    ap.add_argument('--pc-source',default='outputs/experiments')
    ap.add_argument('--server-source',default='analysis_inputs/server/experiments')
    ap.add_argument('--output-dir',default='outputs/analysis_ai_ready')
    ap.add_argument('--class-labels',default=None)
    ap.add_argument('--no-zip',action='store_true')
    args=ap.parse_args()
    root=Path(args.project_root).resolve()
    sources={k: (root/Path(v)).resolve() for k,v in [('pc2',args.pc_source),('server',args.server_source)]}
    output=root/args.output_dir
    result=main(root,sources,output,dataset_scope=args.dataset_scope,seed=args.seed,
                class_labels_file=Path(args.class_labels) if args.class_labels else None,
                create_zip=not args.no_zip)
    print(json.dumps(result,ensure_ascii=False,indent=2))
