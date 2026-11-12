#!/usr/bin/env python3
"""Analyze validated production AID embeddings; never generates/guesses embeddings.

Requires feature_embeddings/<run_id>_final.npz from feature_space_pipeline.py.
Reports all classes, all unordered class pairs, and fair attention comparisons.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ATTENTIONS = ("none", "eca", "cbam", "procbam")
METHODS = ("fetril", "fecam", "ssre", "pass", "efc")
LITERATURE_PAIRS = [
    ("Stadium", "Playground"), ("BareLand", "Desert"), ("Resort", "Park"),
    ("Church", "Commercial"), ("Beach", "Port"), ("Square", "Center"),
    ("MediumResidential", "DenseResidential"),
]
PROVENANCE = ("dataset", "method", "protocol", "backbone", "attention", "seed", "batch", "source_machine", "run_id")


def csv_write(p, df, cols=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if len(df.columns) else pd.DataFrame(columns=cols or [])).to_csv(p, index=False, encoding="utf-8-sig")


def json_write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def read_table(root, zf, name):
    p = root / name
    if p.is_file():
        return pd.read_csv(p)
    if zf is not None and name in zf.namelist():
        return pd.read_csv(io.BytesIO(zf.read(name)))
    return pd.DataFrame()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_embedding(path):
    with np.load(path, allow_pickle=False) as ar:
        required = {"features", "y_true", "y_pred", "task_id", "run_id"}
        if not required.issubset(ar.files):
            raise ValueError(f"Missing required embedding fields {required.difference(ar.files)}")
        F = np.array(ar["features"], dtype=np.float64)
        y = np.array(ar["y_true"], dtype=np.int64)
        pred = np.array(ar["y_pred"], dtype=np.int64)
        sample_id = np.array(ar["sample_id"]).astype(str) if "sample_id" in ar.files else None
        sample_path = np.array(ar["sample_path"]).astype(str) if "sample_path" in ar.files else None
        task_id = int(ar["task_id"])
        run_id = str(ar["run_id"])
    if F.ndim != 2 or F.shape[1] != 512 or F.shape[0] != len(y) or pred.shape != y.shape:
        raise ValueError("Embedding shapes inconsistent or not 512-d")
    if len(y) != 2000 or len(np.unique(y)) != 30 or task_id not in (4, 9):
        raise ValueError("Final AID embedding must have 2000 samples, 30 classes, and task_id 4/9")
    if not np.array_equal(np.unique(y), np.arange(30)):
        raise ValueError("Expected incremental class IDs 0..29")
    if not np.isfinite(F).all() or np.min(np.linalg.norm(F, axis=1)) < 1e-10:
        raise ValueError("Non-finite or zero-norm feature vector")
    return F, y, pred, sample_id, sample_path, task_id, run_id


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return int(ok.sum()), np.nan, np.nan
    try:
        from scipy.stats import spearmanr
        result = spearmanr(x[ok], y[ok])
        return int(ok.sum()), float(result.statistic), float(result.pvalue)
    except ImportError:
        return int(ok.sum()), float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())), np.nan


def true_support_matrix(y, pred, n=30):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def analyze_one(run, path, label_map, forget_lookup, class_accuracy_lookup):
    F, y, pred, sample_ids, sample_paths, task_id, embedding_run_id = load_embedding(path)
    if embedding_run_id != run["run_id"]:
        raise ValueError("run_id mismatch between NPZ and provenance")
    if (run["protocol"] == "T5" and task_id != 4) or (run["protocol"] == "T10" and task_id != 9):
        raise ValueError("Task ID does not match protocol")
    # Normalize each sample then compute centroids in unit feature space.
    X = F / np.linalg.norm(F, axis=1, keepdims=True)
    cm = true_support_matrix(y, pred)
    counts = cm.sum(axis=1)
    centers = np.zeros((30, 512))
    centers_norm = np.zeros_like(centers)
    for cid in range(30):
        if counts[cid] < 1:
            raise ValueError(f"No class support for class {cid}")
        centers[cid] = np.mean(X[y == cid], axis=0)
        center_norm = np.linalg.norm(centers[cid])
        if center_norm < 1e-10:
            raise ValueError(f"Centroid near-zero norm class {cid}")
        centers_norm[cid] = centers[cid] / center_norm
    S = centers_norm @ centers_norm.T
    if np.min(S) < -1-1e-6 or np.max(S) > 1+1e-6 or not np.isfinite(S).all():
        raise ValueError("Invalid cosine similarity values")
    sim_rows, class_rows, pair_rows, sample_rows = [], [], [], []
    meta = {k: run.get(k) for k in PROVENANCE}
    if len(sample_ids) if sample_ids is not None else 0:
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("Duplicate sample IDs")
    for cid in range(30):
        mask = y == cid
        d = 1 - X[mask] @ centers_norm[cid]
        if np.min(d) < -1e-5 or np.max(d) > 2+1e-5:
            raise ValueError("Invalid cosine distance")
        label = label_map[cid]
        record = {**meta, "class_id": cid, "class_name": label, "support": int(counts[cid]),
                  "mean_cosine_distance_to_centroid": float(np.mean(d)),
                  "median_cosine_distance_to_centroid": float(np.median(d)),
                  "std_cosine_distance_to_centroid": float(np.std(d, ddof=0)),
                  "min": float(np.min(d)), "max": float(np.max(d)),
                  "p75": float(np.percentile(d, 75)), "p90": float(np.percentile(d, 90)),
                  "p95": float(np.percentile(d, 95)),
                  "final_class_accuracy_pct": float(cm[cid, cid] / counts[cid] * 100),
                  "final_class_error_rate_pct": float(100 - cm[cid, cid] / counts[cid] * 100),
                  "class_forgetting_pp": forget_lookup.get((run["run_id"], cid), np.nan)}
        key = (run["run_id"], cid)
        reported = class_accuracy_lookup.get(key)
        if reported is not None and abs(reported - record["final_class_accuracy_pct"]) > 0.051:
            raise ValueError(f"Canonical final class accuracy mismatch for {label}: {reported} vs {record['final_class_accuracy_pct']}")
        class_rows.append(record)
        inds = np.where(mask)[0]
        for ix, dist in zip(inds, d):
            sample_rows.append({"run_id": run["run_id"], "class_id": cid, "class_name": label,
                                "sample_id": sample_ids[ix] if sample_ids is not None else "",
                                "sample_path": sample_paths[ix] if sample_paths is not None else "",
                                "cosine_distance_to_centroid": float(dist), "sample_index": int(ix)})
    for i in range(30):
        for j in range(i+1, 30):
            sim = float(S[i, j])
            base = {**meta, "class_i_id": i, "class_i_name": label_map[i],
                    "class_j_id": j, "class_j_name": label_map[j],
                    "support_i": int(counts[i]), "support_j": int(counts[j])}
            sim_rows.append({**base, "centroid_cosine_similarity": sim,
                             "centroid_cosine_distance": 1 - sim})
            ij = int(cm[i, j]); ji = int(cm[j, i]); support = int(counts[i] + counts[j])
            pair_rows.append({**base, "centroid_cosine_similarity": sim,
                              "error_i_to_j": ij, "error_j_to_i": ji,
                              "symmetric_confusion_count": ij + ji,
                              "symmetric_confusion_rate_pct": (ij + ji) / support * 100})
    return (pd.DataFrame(sim_rows), pd.DataFrame(class_rows), pd.DataFrame(pair_rows),
            pd.DataFrame(sample_rows), S, cm)


def plot_one(run_id, S, sims, classes, pairs, samples, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label_names = [classes[classes.class_id == i].class_name.iloc[0] for i in range(30)]
    # 1: full centroid similarity heatmap.
    fig, ax = plt.subplots(figsize=(12, 10))
    masked = np.ma.masked_where(np.eye(30, dtype=bool), S)
    im = ax.imshow(masked, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(30), label_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(30), label_names, fontsize=6)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(f"Centroid similarity — {run_id}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"centroid_similarity_heatmap_{run_id}.png", dpi=300)
    plt.close(fig)
    # 2: dispersion ranking.
    data = classes.sort_values("mean_cosine_distance_to_centroid")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(data.class_name, data.mean_cosine_distance_to_centroid)
    ax.set_xlabel("Mean cosine distance to class centroid")
    ax.set_title(f"Intra-class dispersion — {run_id}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"intra_class_dispersion_{run_id}.png", dpi=300)
    plt.close(fig)
    # 3: similarity vs symmetric confusion.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pairs.centroid_cosine_similarity, pairs.symmetric_confusion_rate_pct, alpha=.5, s=14)
    ax.set_xlabel("Centroid cosine similarity")
    ax.set_ylabel("Symmetric confusion rate (%)")
    ax.set_title(f"Similarity vs confusion — {run_id}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"similarity_vs_confusion_{run_id}.png", dpi=300)
    plt.close(fig)
    # 4: dispersion vs error rate.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(classes.mean_cosine_distance_to_centroid, classes.final_class_error_rate_pct)
    for row in classes.sort_values("final_class_error_rate_pct", ascending=False).head(5).itertuples():
        ax.annotate(row.class_name, (row.mean_cosine_distance_to_centroid, row.final_class_error_rate_pct), fontsize=7)
    ax.set_xlabel("Mean intra-class cosine dispersion")
    ax.set_ylabel("Final class error rate (%)")
    ax.set_title(f"Dispersion vs error — {run_id}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"dispersion_vs_error_{run_id}.png", dpi=300)
    plt.close(fig)
    # 5: target pairs supplied in project research text (not proof of visual similarity).
    targets = {frozenset([norm(a), norm(b)]): f"{a} ↔ {b}" for a, b in LITERATURE_PAIRS}
    target = sims.copy()
    target["key"] = [frozenset([norm(a), norm(b)]) for a, b in zip(target.class_i_name,target.class_j_name)]
    target["pair"] = target.key.map(lambda k: targets.get(k))
    target = target[target.pair.notna()].sort_values("centroid_cosine_similarity")
    if not target.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(target.pair, target.centroid_cosine_similarity)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Centroid cosine similarity (unitless)")
        ax.set_title(f"Preselected class pairs — {run_id}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"literature_pairs_similarity_{run_id}.png", dpi=300)
        plt.close(fig)
    # 6: distance distribution for top-five dispersion classes (objective selection).
    wanted = classes.nlargest(5,"mean_cosine_distance_to_centroid").class_id.to_list()
    subset = samples[samples.class_id.isin(wanted)]
    if not subset.empty:
        vals = [subset[subset.class_id==k].cosine_distance_to_centroid.to_numpy(float) for k in wanted]
        names = [classes[classes.class_id==k].class_name.iloc[0] for k in wanted]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(vals, labels=names)
        ax.set_ylabel("Cosine distance to centroid")
        ax.set_title(f"Selected intra-class distance distributions — {run_id}")
        plt.xticks(rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"selected_class_distance_boxplot_{run_id}.png", dpi=300)
        plt.close(fig)


def plot_attention_effects(pair_delta, class_delta, summary, out, n=4):
    """Objective 4-case selection: confusion reduction, similarity reduction,
    compactness improvement, and AIA degradation. No manual cherry-picking."""
    if summary.empty or pair_delta.empty or class_delta.empty or n <= 0:
        return pd.DataFrame()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    rules = [
        ("largest_mutual_confusion_reduction", "mean_delta_mutual_confusion_pp"),
        ("largest_similarity_reduction", "mean_delta_inter_class_similarity"),
        ("largest_compactness_improvement", "mean_delta_intra_dispersion"),
        ("largest_aia_degradation", "delta_aia_pp"),
    ]
    choices = []
    used = set()
    for label, col in rules:
        if col not in summary:
            continue
        for r in summary.sort_values(col, ascending=True).itertuples(index=False):
            key = (r.baseline_run_id, r.attention_run_id)
            if key not in used:
                choices.append({"selection_criterion":label, "baseline_run_id":key[0],
                                "attention_run_id":key[1], "selection_metric":col,
                                "selection_value":getattr(r,col)})
                used.add(key)
                break
        if len(choices) >= n:
            break
    for ch in choices:
        base_id=ch["baseline_run_id"]; att_id=ch["attention_run_id"]
        z=pair_delta[(pair_delta.baseline_run_id==base_id)&(pair_delta.attention_run_id==att_id)]
        w=class_delta[(class_delta.baseline_run_id==base_id)&(class_delta.attention_run_id==att_id)]
        if z.empty or w.empty:
            continue
        mat=np.zeros((30,30),dtype=float)
        for r in z.itertuples(index=False):
            mat[int(r.class_i_id),int(r.class_j_id)]=r.delta_similarity
            mat[int(r.class_j_id),int(r.class_i_id)]=r.delta_similarity
        lim=max(0.01,float(np.max(np.abs(mat))))
        fig,ax=plt.subplots(figsize=(10,8))
        im=ax.imshow(np.ma.masked_where(np.eye(30,dtype=bool),mat),cmap="RdBu_r",vmin=-lim,vmax=lim)
        ax.set_title(f"Δ centroid similarity — {att_id}")
        ax.set_xlabel('Predicted class index is NOT used here; class j')
        ax.set_ylabel('Class i')
        fig.colorbar(im,ax=ax,label='Δ cosine similarity (unitless)')
        fig.tight_layout()
        fig.savefig(fig_dir/f"attention_delta_similarity_heatmap_{att_id}.png",dpi=300)
        plt.close(fig)
        w=w.sort_values('delta_intra_dispersion')
        fig,ax=plt.subplots(figsize=(10,9))
        ax.barh(w.class_name,w.delta_intra_dispersion)
        ax.axvline(0,color='black',linewidth=.8)
        ax.set_xlabel('Δ mean intra-class cosine dispersion (unitless)')
        ax.set_title(f"Δ class compactness — {att_id}")
        fig.tight_layout()
        fig.savefig(fig_dir/f"attention_delta_dispersion_{att_id}.png",dpi=300)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(summary.mean_delta_inter_class_similarity, summary.mean_delta_intra_dispersion)
    ax.axhline(0,color='gray',linewidth=.7)
    ax.axvline(0,color='gray',linewidth=.7)
    ax.set_xlabel('Mean Δinter-class centroid cosine similarity')
    ax.set_ylabel('Mean Δintra-class cosine dispersion')
    ax.set_title('Attention feature effects across fair pairs')
    fig.tight_layout()
    fig.savefig(fig_dir/'attention_feature_effect_summary.png',dpi=300)
    plt.close(fig)
    return pd.DataFrame(choices)


def choose_contact_plan(samples, out, seed=1993):
    if samples.empty:
        return pd.DataFrame()
    rows = []
    for run_id, group in samples.groupby("run_id"):
        for class_id, d in group.groupby("class_id"):
            d = d.sort_values("cosine_distance_to_centroid").reset_index(drop=True)
            if d.empty:
                continue
            for label, q in [("closest", 0), ("median", .5), ("p75", .75), ("p90", .90), ("farthest", 1)]:
                sel = d.iloc[int(round(q * (len(d)-1)))]
                rows.append({"run_id": run_id, "class_id": class_id,
                             "class_name": sel.class_name, "selection": label,
                             "sample_index": int(sel.sample_index), "sample_id": sel.sample_id,
                             "sample_path": sel.sample_path,
                             "cosine_distance_to_centroid": float(sel.cosine_distance_to_centroid)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--feature-dir", default="outputs/feature_analysis/feature_embeddings")
    ap.add_argument("--output-dir", default="outputs/feature_analysis")
    ap.add_argument("--ai-dir", default="outputs/analysis_ai_ready")
    ap.add_argument("--ai-bundle", default=None)
    ap.add_argument("--plot-max-runs", type=int, default=5, help="0 to skip, 80 for all")
    ap.add_argument("--plot-max-pairs", type=int, default=4, help="Up to four objectively selected attention pairs")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    embed_dir = root / args.feature_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ai_dir = root / args.ai_dir
    ai_zip = Path(args.ai_bundle) if args.ai_bundle else ai_dir / "ai_ready_evaluation_bundle.zip"
    zf = zipfile.ZipFile(ai_zip) if ai_zip.exists() else None
    labels_df = read_table(ai_dir, zf, "class_label_map.csv")
    models = read_table(ai_dir, zf, "model_insights_all.csv")
    canonical_classes = read_table(ai_dir, zf, "class_accuracy_long.csv")
    forgetting = read_table(ai_dir, zf, "class_forgetting.csv")
    fair_pairs = read_table(ai_dir, zf, "paired_attention_effects.csv")
    if zf:
        zf.close()
    if labels_df.empty or models.empty:
        raise RuntimeError("Need class_label_map.csv and model_insights_all.csv (AI-ready v2.1 production export)")
    m = labels_df[labels_df.dataset == "aid"].copy()
    if len(m) != 30 or m["incremental_id"].nunique() != 30 or not m["class_label_verified"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("AID incremental class name mapping incomplete/unverified")
    label_map = dict(zip(m.incremental_id.astype(int), m.class_name.astype(str)))
    if any(norm(s).startswith("c") and re.fullmatch(r"c\d+", norm(s)) for s in label_map.values()):
        raise RuntimeError("Dummy Cxx class mapping is not allowed for feature analysis")
    # Class accuracy and forgetting only serve cross-check and context; do not redefine official metrics.
    lookup_forget = {}
    if not forgetting.empty:
        lookup_forget = {(str(r.run_id), int(r.class_id)): float(r.class_forgetting_pp)
                         for r in forgetting.itertuples(index=False)}
    lookup_accuracy = {}
    if not canonical_classes.empty:
        c = canonical_classes[canonical_classes.task_id == canonical_classes.protocol.map({"T5": 4, "T10": 9})]
        lookup_accuracy = {(str(r.run_id), int(r.class_id)): float(r.class_accuracy_pct)
                           for r in c.itertuples(index=False)}
    all_sim, all_class, all_pairs, all_samples, errors, validations = [], [], [], [], [], []
    matrix_cache = {}
    alignment_by_run = {}
    for r in models.to_dict("records"):
        if str(r.get("dataset")) != "aid" or str(r.get("ready")).lower() not in ("true", "1"):
            continue
        run_id = str(r["run_id"])
        p = embed_dir / f"{run_id}_final.npz"
        if not p.is_file():
            errors.append({"run_id": run_id, "status": "missing_embedding", "message": str(p)})
            continue
        run = dict(r)
        run["dataset"] = "aid"
        try:
            sims, classes, pairs, samples, S, cm = analyze_one(run, p, label_map, lookup_forget, lookup_accuracy)
            all_sim.append(sims); all_class.append(classes); all_pairs.append(pairs); all_samples.append(samples)
            matrix_cache[run_id] = (S, cm, sims, classes, pairs)
            # Record exact canonical order and optional stable sample identity for pair validation.
            with np.load(p, allow_pickle=False) as emb:
                alignment_by_run[run_id] = (
                    np.asarray(emb["y_true"], dtype=np.int64),
                    np.asarray(emb["sample_id"]).astype(str) if "sample_id" in emb.files else None,
                )
            validations.append({"run_id": run_id, "status": "validated", "feature_dim": 512, "n_samples": 2000})
        except Exception as exc:
            errors.append({"run_id": run_id, "status": "invalid_embedding", "message": f"{type(exc).__name__}: {exc}"})
    def join(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sims, classes, pairs, samples = map(join, (all_sim, all_class, all_pairs, all_samples))
    top = (sims.sort_values(["run_id", "centroid_cosine_similarity"], ascending=[True, False])
             .groupby("run_id", group_keys=False).head(20).copy()) if not sims.empty else pd.DataFrame()
    if not top.empty:
        top["similarity_rank"] = top.groupby("run_id").cumcount() + 1
    correlation_rows = []
    for run_id, grp in pairs.groupby("run_id") if not pairs.empty else []:
        n, rho, pval = spearman(grp.centroid_cosine_similarity, grp.symmetric_confusion_rate_pct)
        correlation_rows.append({"run_id": run_id, "analysis_type": "similarity_vs_symmetric_confusion", "n": n, "spearman_rho": rho, "p_value": pval})
    for run_id, grp in classes.groupby("run_id") if not classes.empty else []:
        for var, typ in [("final_class_error_rate_pct", "dispersion_vs_final_error"),
                         ("class_forgetting_pp", "dispersion_vs_class_forgetting")]:
            n, rho, pval = spearman(grp.mean_cosine_distance_to_centroid, grp[var])
            correlation_rows.append({"run_id": run_id, "analysis_type": typ, "n": n, "spearman_rho": rho, "p_value": pval})
    correlations = pd.DataFrame(correlation_rows)
    pair_deltas, class_deltas, literature = [], [], []
    fairset = set()
    if not fair_pairs.empty:
        fairset = {(str(r.baseline_run_id), str(r.attention_run_id)) for r in fair_pairs.itertuples(index=False)}
    fair_lookup = {(str(r.baseline_run_id), str(r.attention_run_id)): r for r in fair_pairs.itertuples(index=False)} if not fair_pairs.empty else {}
    for base_id, att_id in sorted(fairset):
        if base_id not in matrix_cache or att_id not in matrix_cache:
            continue
        fp = fair_lookup[(base_id, att_id)]
        pair_meta = {"dataset": fp.dataset, "method": fp.method, "protocol": fp.protocol,
                     "backbone": fp.backbone, "attention": fp.attention, "seed": int(fp.seed),
                     "batch": int(fp.batch), "machine": fp.source_machine}
        y0, ids0 = alignment_by_run[base_id]
        y1, ids1 = alignment_by_run[att_id]
        if not np.array_equal(y0, y1):
            errors.append({"run_id": att_id, "status": "paired_y_true_order_mismatch", "message": base_id})
            continue
        if ids0 is not None and ids1 is not None and not np.array_equal(ids0, ids1):
            errors.append({"run_id": att_id, "status": "paired_sample_ids_mismatch", "message": base_id})
            continue
        base = matrix_cache[base_id]; att = matrix_cache[att_id]
        bsim, asim = base[2], att[2]
        bpair, apair = base[4], att[4]
        bclass, aclass = base[3], att[3]
        # All class IDs are from one verified manifest; same test support is mandatory.
        keys = ["class_i_id", "class_j_id"]
        z = bpair.merge(apair, on=keys, suffixes=("_none", "_attention"), validate="one_to_one")
        if not np.array_equal(z.support_i_none.to_numpy(), z.support_i_attention.to_numpy()) or not np.array_equal(z.support_j_none.to_numpy(), z.support_j_attention.to_numpy()):
            errors.append({"run_id": att_id, "status": "pair_support_mismatch", "message": base_id})
            continue
        for r in z.itertuples(index=False):
            pair_deltas.append({
                **pair_meta, "row_type": "class_pair", "baseline_run_id": base_id, "attention_run_id": att_id,
                "class_i_id": int(r.class_i_id), "class_j_id": int(r.class_j_id),
                "class_i_name": r.class_i_name_none, "class_j_name": r.class_j_name_none,
                "baseline_centroid_similarity": float(r.centroid_cosine_similarity_none),
                "attention_centroid_similarity": float(r.centroid_cosine_similarity_attention),
                "delta_similarity": float(r.centroid_cosine_similarity_attention-r.centroid_cosine_similarity_none),
                "baseline_mutual_confusion_pct": float(r.symmetric_confusion_rate_pct_none),
                "attention_mutual_confusion_pct": float(r.symmetric_confusion_rate_pct_attention),
                "delta_mutual_confusion_pp": float(r.symmetric_confusion_rate_pct_attention-r.symmetric_confusion_rate_pct_none),
            })
        zc = bclass.merge(aclass, on="class_id", suffixes=("_none", "_attention"), validate="one_to_one")
        if not np.array_equal(zc.support_none, zc.support_attention):
            errors.append({"run_id": att_id, "status": "class_support_mismatch", "message": base_id})
            continue
        for r in zc.itertuples(index=False):
            class_deltas.append({**pair_meta, "row_type": "class", "baseline_run_id": base_id,
                                  "attention_run_id": att_id, "class_id": int(r.class_id),
                                  "class_name": r.class_name_none,
                                  "baseline_intra_dispersion": float(r.mean_cosine_distance_to_centroid_none),
                                  "attention_intra_dispersion": float(r.mean_cosine_distance_to_centroid_attention),
                                  "delta_intra_dispersion": float(r.mean_cosine_distance_to_centroid_attention-r.mean_cosine_distance_to_centroid_none)})
    pair_del = pd.DataFrame(pair_deltas)
    class_del = pd.DataFrame(class_deltas)
    canon = {frozenset([norm(a), norm(b)]): f"{a} ↔ {b}" for a, b in LITERATURE_PAIRS}
    lit_frames = []
    # Baseline records are important even if none of the attention checkpoints is extractable.
    if not pairs.empty:
        base_lit = pairs[pairs.attention == "none"].copy()
        if not base_lit.empty:
            base_lit["pair_norm"] = [frozenset([norm(a), norm(b)]) for a,b in zip(base_lit.class_i_name,base_lit.class_j_name)]
            base_lit["class_pair"] = base_lit.pair_norm.map(lambda v: canon.get(v))
            base_lit = base_lit[base_lit.class_pair.notna()].copy()
            base_lit = base_lit.rename(columns={"centroid_cosine_similarity":"baseline_centroid_similarity",
                                                "symmetric_confusion_rate_pct":"baseline_mutual_confusion_pct",
                                                "run_id":"baseline_run_id"})
            base_lit["attention_run_id"] = pd.NA
            base_lit["attention_centroid_similarity"] = np.nan
            base_lit["attention_mutual_confusion_pct"] = np.nan
            base_lit["delta_similarity"] = np.nan
            base_lit["delta_mutual_confusion_pp"] = np.nan
            lit_frames.append(base_lit.drop(columns="pair_norm"))
    if not pair_del.empty:
        p = pair_del.copy()
        p["pair_norm"] = [frozenset([norm(a), norm(b)]) for a, b in zip(p.class_i_name, p.class_j_name)]
        p["class_pair"] = p.pair_norm.map(lambda v: canon.get(v))
        lit_frames.append(p[p.class_pair.notna()].drop(columns="pair_norm").copy())
    literature = pd.concat(lit_frames, ignore_index=True) if lit_frames else pd.DataFrame()
    pair_del = pair_del if not pair_del.empty else pd.DataFrame()
    # Feature-effect aggregate: disjoint scalar summaries per fair baseline-attention pair, no vector coordinate pooling.
    summary = pd.DataFrame()
    if not pair_del.empty and not class_del.empty:
        s1 = pair_del.groupby(["baseline_run_id", "attention_run_id"], as_index=False).agg(
            mean_delta_inter_class_similarity=("delta_similarity", "mean"),
            mean_delta_mutual_confusion_pp=("delta_mutual_confusion_pp", "mean"), n_class_pairs=("class_i_id", "size"))
        s2 = class_del.groupby(["baseline_run_id", "attention_run_id"], as_index=False).agg(
            mean_delta_intra_dispersion=("delta_intra_dispersion", "mean"), n_classes=("class_id", "size"))
        summary = s1.merge(s2, on=["baseline_run_id", "attention_run_id"], validate="one_to_one")
        keep = ["baseline_run_id", "attention_run_id", "dataset", "method", "protocol", "backbone",
                "attention", "seed", "batch", "source_machine", "delta_aia_pp", "delta_lta_pp"]
        if not fair_pairs.empty:
            summary = summary.merge(fair_pairs[keep], on=["baseline_run_id", "attention_run_id"], how="left", validate="one_to_one")
    if args.plot_max_runs:
        # Select deterministically: baseline first, then run ID; no subjective cherry-picking.
        ordered = sorted(matrix_cache, key=lambda r: (not r.endswith("_none_seed1993"), r))[:args.plot_max_runs]
        for run_id in ordered:
            plot_one(run_id, matrix_cache[run_id][0], matrix_cache[run_id][2],
                     matrix_cache[run_id][3], matrix_cache[run_id][4],
                     samples[samples.run_id == run_id], out)
    selected_cases = plot_attention_effects(pair_del, class_del, summary, out, args.plot_max_pairs)
    plan = choose_contact_plan(samples, out)
    outputs = {
        "feature_representative_cases.csv": selected_cases,
        "inter_class_similarity.csv": sims,
        "inter_class_similarity_top_pairs.csv": top,
        "intra_class_dispersion.csv": classes,
        "class_feature_statistics.csv": classes,
        "feature_confusion_join.csv": pairs,
        "feature_correlation_summary.csv": correlations,
        "attention_feature_delta.csv": pd.concat([pair_del, class_del], ignore_index=True) if not pair_del.empty or not class_del.empty else pd.DataFrame(),
        "attention_feature_summary.csv": summary,
        "literature_class_pairs_analysis.csv": literature,
        "feature_contact_sheet_selection.csv": plan,
        "feature_sample_distances.csv": samples,
        "feature_analysis_errors.csv": pd.DataFrame(errors),
        "feature_analysis_validated_runs.csv": pd.DataFrame(validations),
        "table_inter_class_literature_pairs.csv": literature,
        "table_intra_class_top.csv": (classes.sort_values("mean_cosine_distance_to_centroid", ascending=False).groupby("run_id",group_keys=False).head(10) if not classes.empty else pd.DataFrame()),
        "table_attention_feature_summary.csv": summary,
    }
    for name, df in outputs.items():
        csv_write(out / name, df)
    insights = []
    for run_id in sorted(matrix_cache):
        sr = top[top.run_id == run_id].head(1)
        cr = classes[classes.run_id == run_id].nlargest(1, "mean_cosine_distance_to_centroid")
        if sr.empty or cr.empty:
            continue
        best = sr.iloc[0]; spread = cr.iloc[0]
        insights.append({"run_id": run_id, "evidence": [
            {"filename": "inter_class_similarity.csv", "columns": ["centroid_cosine_similarity","class_i_name","class_j_name"],
             "numeric_value": float(best.centroid_cosine_similarity), "class_pair": [str(best.class_i_name), str(best.class_j_name)]},
            {"filename": "intra_class_dispersion.csv", "columns": ["mean_cosine_distance_to_centroid","class_name"],
             "numeric_value": float(spread.mean_cosine_distance_to_centroid), "class_name": str(spread.class_name)}],
            "top_inter_class_pairs": [
                {"class_i":str(rr.class_i_name), "class_j":str(rr.class_j_name),
                 "centroid_cosine_similarity":float(rr.centroid_cosine_similarity)}
                for rr in top[top.run_id == run_id].itertuples(index=False)],
            "top_dispersion_classes": [
                {"class_name":str(rr.class_name),
                 "mean_cosine_distance_to_centroid":float(rr.mean_cosine_distance_to_centroid)}
                for rr in classes[classes.run_id == run_id].nlargest(5,"mean_cosine_distance_to_centroid").itertuples(index=False)],
            "caveat": "Descriptive model-specific feature-space geometry, not causal or visual similarity proof."})
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset":"aid", "seed":1993,
                "expected_baselines":20,"expected_runs_all":80,
                "requested_run_ids": models.loc[models.dataset == "aid", "run_id"].dropna().tolist(),
                "extracted_run_ids": sorted(matrix_cache),
                "missing_or_invalid_run_ids": [str(r["run_id"]) for r in errors],
                "input_directories": {"embedding":str(embed_dir), "ai_ready":str(ai_dir)},
                "validated_embedded_runs":len(matrix_cache), "missing_or_invalid":len(errors),
                "n_fair_feature_attention_pairs":len(summary),
                "feature_dims":sorted(set([512] if matrix_cache else [])),
                "files_generated":sorted(outputs),
                "figures_generated":sorted(p.name for p in (out / "figures").glob("*.png")) if (out / "figures").exists() else [],
                "cautions":["Similarity is not classification accuracy.",
                           "Cosine similarity/distance are dimensionless; not percentage points.",
                           "Confusion and accuracy deltas are percentage points.",
                           "Single seed; Spearman p-values are descriptive and observations are dependent.",
                           "Mean ΔAIA across four conditions is not four independent seeds.",
                           "No cross-model feature-coordinate alignment is assumed."]}
    json_write(out / "feature_analysis_manifest.json", manifest)
    json_write(out / "feature_analysis_data_dictionary.json", {
        "metrics": {
            "centroid_cosine_similarity": "Cosine between within-run unit-normalized class centroids; [-1,1], unitless",
            "mean_cosine_distance_to_centroid": "Mean 1-cos(feature, own class centroid), unitless",
            "symmetric_confusion_rate_pct": "(error_i_to_j + error_j_to_i)/(support_i+support_j)*100, percent",
            "delta_similarity": "attention minus none, unitless",
            "delta_intra_dispersion": "attention minus none, unitless",
            "delta_mutual_confusion_pp": "attention minus none, percentage points",
            "class_forgetting_pp": "Read from existing AI-ready exporter; not recomputed",
        }, "warnings": manifest["cautions"]})
    comprehensive_attention = []
    if not summary.empty:
        for r in summary.to_dict("records"):
            comprehensive_attention.append({
                "baseline_run_id": r["baseline_run_id"],
                "attention_run_id": r["attention_run_id"],
                "values": {k: r.get(k) for k in (
                    "mean_delta_inter_class_similarity", "mean_delta_intra_dispersion",
                    "mean_delta_mutual_confusion_pp", "delta_aia_pp", "delta_lta_pp")},
                "evidence_file": "attention_feature_summary.csv",
                "evidence_columns": ["baseline_run_id", "attention_run_id",
                    "mean_delta_inter_class_similarity", "mean_delta_intra_dispersion",
                    "mean_delta_mutual_confusion_pp", "delta_aia_pp", "delta_lta_pp"],
                "caveat": "Within-run geometry comparisons only; no cross-model coordinate alignment, no causality or single-seed significance."})
    json_write(out / "feature_analysis_insights_all.json", {"coverage":manifest,
        "per_run_insights":insights,
        "per_attention_pair_insights":comprehensive_attention,
        "literature_class_pairs_file":"literature_class_pairs_analysis.csv",
        "descriptive_correlations_file":"feature_correlation_summary.csv",
        "all_class_pairs_csv":"inter_class_similarity.csv",
        "all_classes_csv":"intra_class_dispersion.csv",
        "attention_pairs_csv":"attention_feature_summary.csv"})
    json_write(out / "feature_analysis_insights.json", {"count":len(insights),"highlight":insights[:5],"all_file":"feature_analysis_insights_all.json"})
    print(json.dumps({"validated_runs":len(matrix_cache),"errors":len(errors),
                      "fair_feature_pairs":len(summary),"output_dir":str(out)}, indent=2))

if __name__ == "__main__":
    main()
