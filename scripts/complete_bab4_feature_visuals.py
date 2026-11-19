#!/usr/bin/env python3
"""Complete reproducible feature-space visuals without retraining or checkpoint loading.

Inputs: AI-ready feature-space ZIP (contains feature_contact_sheet_selection.csv and
feature_confusion_join.csv), and optionally the local original AID image directory.

This script NEVER fabricates image tiles. Missing files fail with a clear error.
All samples are selected from ONE explicitly named baseline run and traceable by
sample_id. For contact sheets, selections are feature-distance quantiles, not
an assertion about image visual similarity/variability.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

INTER_PAIRS = (("Desert", "BareLand"), ("Stadium", "Playground"), ("Resort", "Park"))
INTRA_CLASSES = ("Forest", "Desert", "Park")
SELECTIONS = ("closest", "median", "p90", "farthest")
DEFAULT_RUN = "aid_efc_t5_r18_none_seed1993"
MISSING_SCATTERS = (
    "aid_efc_t10_r18_none_seed1993",
    "aid_efc_t10_r34_none_seed1993",
    "aid_efc_t5_r34_none_seed1993",
    "aid_fecam_t10_r18_none_seed1993",
)
LITERATURE_PAIRS = {
    frozenset(pair) for pair in (
        ("Desert", "BareLand"), ("Stadium", "Playground"),
        ("Resort", "Park"), ("Church", "Commercial"),
        ("Beach", "Port"), ("Square", "Center"),
        ("MediumResidential", "DenseResidential"),
    )
}


def read_csv(source_zip: Path, name: str, selection_csv: Path | None = None):
    import pandas as pd
    if selection_csv and name == "feature_contact_sheet_selection.csv":
        if not selection_csv.is_file():
            raise FileNotFoundError(selection_csv)
        return pd.read_csv(selection_csv, keep_default_na=False)
    with zipfile.ZipFile(source_zip) as z:
        paths = [f"feature_space/{name}", name]
        found = next((x for x in paths if x in z.namelist()), None)
        if not found:
            raise FileNotFoundError(f"{name} not found in {source_zip}")
        return pd.read_csv(io.BytesIO(z.read(found)), keep_default_na=False)


def resolve_aid_sample(root: Path, sample_id: str) -> Path:
    """Never use stale absolute sample_path; require safe relative sample_id."""
    raw = sample_id.replace('\\', '/')
    rel = PurePosixPath(raw)
    if rel.is_absolute() or any(part in {'.', '..', ''} for part in rel.parts) or len(rel.parts) < 2:
        raise ValueError(f"Unsafe/invalid sample_id: {sample_id!r}")
    path = root.joinpath(*rel.parts)
    if not path.is_file():
        # Resolve case differences deterministically, forbidding ambiguity.
        node = root
        for part in rel.parts:
            if not node.is_dir():
                raise FileNotFoundError(f"Missing dataset directory for {sample_id}: {node}")
            candidates = sorted(p for p in node.iterdir() if p.name.casefold() == part.casefold())
            if len(candidates) != 1:
                raise FileNotFoundError(f"Could not resolve uniquely under AID root: {sample_id} ({node})")
            node = candidates[0]
        path = node
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Sample escapes AID root: {sample_id}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def get_font(size: int):
    from PIL import ImageFont
    for family in ('DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(family, size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_tile(path: Path, class_name: str, selection: str, distance: float,
               tile_size: int = 202, height: int = 264):
    from PIL import Image, ImageDraw, ImageOps
    result = Image.new('RGB', (tile_size, height), 'white')
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        fitted = ImageOps.contain(im, (tile_size - 12, 198))
        result.paste(fitted, ((tile_size - fitted.width)//2, (198 - fitted.height)//2 + 5))
    d = ImageDraw.Draw(result)
    d.text((8, 209), class_name, font=get_font(17), fill='#111111')
    d.text((8, 232), f"{selection}  d={distance:.4f}", font=get_font(13), fill='#333333')
    return result


def contact_sheet(args):
    from PIL import Image, ImageDraw
    src = args.source_zip
    if not args.aid_root:
        raise ValueError('--aid-root is REQUIRED for contact sheets (the original AID images)')
    root = args.aid_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Original AID directory not found: {root}")
    df = read_csv(src, 'feature_contact_sheet_selection.csv', args.selection_csv)
    required = {'run_id','class_name','selection','sample_id','cosine_distance_to_centroid'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in contact selections: {sorted(required-set(df.columns))}")
    df = df.loc[df['run_id'].eq(args.run_id)].copy()
    if len(df) != 150:
        raise ValueError(f"Expected 30 classes x 5 samples = 150 selections in selected run, got {len(df)}")
    selections = {}
    names = {name for pair in INTER_PAIRS for name in pair} | set(INTRA_CLASSES)
    paths = []
    for name in sorted(names):
        for sel in SELECTIONS:
            found = df[(df.class_name == name) & (df.selection == sel)]
            if len(found) != 1:
                raise ValueError(f"Expected 1 candidate: {args.run_id} / {name} / {sel}; found {len(found)}")
            r = found.iloc[0]
            sid = str(r.sample_id)
            p = resolve_aid_sample(root, sid)
            dist = float(r.cosine_distance_to_centroid)
            if not (-1e-5 <= dist <= 2 + 1e-5):
                raise ValueError(f"Invalid cosine distance: {dist}")
            selections[(name,sel)] = (p, sid, dist)
            paths.append(p)
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate original images found across selected tiles; review sample selection')

    target = args.output_dir.resolve()
    target.mkdir(parents=True,exist_ok=True)
    tile_w, tile_h, gap, left, top = 202, 264, 12, 26, 95
    chosen = []
    def pick(name, sel, panel):
        path, sid, dist = selections[(name,sel)]
        chosen.append({
            'run_id':args.run_id, 'panel':panel, 'class_name':name,
            'selection':sel,'sample_id':sid,
            'cosine_distance_to_centroid':dist,
            'sha256_image':hashlib.sha256(path.read_bytes()).hexdigest()
        })
        return image_tile(path,name,sel,dist,tile_w,tile_h)

    # Inter-class: each 4-tile class block, paired class groups in three rows.
    row_h = tile_h + 76
    width = left*2 + (8*tile_w + 7*gap)
    height = top + 3*row_h + 16
    canvas = Image.new('RGB',(width,height),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((left,12), f'Inter-class AID examples - {args.run_id}',font=get_font(23),fill='#111111')
    draw.text((left,46),'Deterministic feature-distance selections; NOT direct visual-similarity measurements',
              font=get_font(15),fill='#555555')
    for ir,(a,b) in enumerate(INTER_PAIRS):
        y = top + ir*row_h
        draw.text((left,y-29),f'{a}  vs  {b}',font=get_font(20),fill='#111111')
        for j,name in enumerate((a,b)):
            for k,sel in enumerate(SELECTIONS):
                x=left+(j*4+k)*(tile_w+gap)
                canvas.paste(pick(name,sel,'inter'),(x,y))
    inter_path = target/'inter_class_contact_sheet.png'
    canvas.save(inter_path,dpi=(300,300))

    # Intra-class: predefined illustrative classes; not claimed to be the top-3 dispersion classes.
    width = left*2 + 4*tile_w + 3*gap
    height = top + 3*row_h + 16
    canvas = Image.new('RGB',(width,height),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((left,12),f'Intra-class AID examples - {args.run_id}',font=get_font(22),fill='#111111')
    draw.text((left,46),'Examples: Forest, Desert, Park | selected by distance to within-run class centroid',
              font=get_font(13),fill='#555555')
    for ir,name in enumerate(INTRA_CLASSES):
        y=top+ir*row_h
        draw.text((left,y-29),name,font=get_font(20),fill='#111111')
        for k,sel in enumerate(SELECTIONS):
            x=left+k*(tile_w+gap)
            canvas.paste(pick(name,sel,'intra'),(x,y))
    intra_path=target/'intra_class_contact_sheet.png'
    canvas.save(intra_path,dpi=(300,300))

    csv_path=target/'contact_sheet_selected_samples.csv'
    with csv_path.open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(chosen[0]))
        writer.writeheader();writer.writerows(chosen)
    summary = {
        'status':'passed','source_run':args.run_id,'selection_rule':'closest,median,p90,farthest based on feature-space cosine distance',
        'inter_class_pairs':list(INTER_PAIRS),'intra_class_illustrative_classes':list(INTRA_CLASSES),
        'tiles_inter':len(INTER_PAIRS)*2*len(SELECTIONS),'tiles_intra':len(INTRA_CLASSES)*len(SELECTIONS),
        'images':[str(inter_path),str(intra_path)],'metadata_csv':str(csv_path),
        'important':'Original AID test-set images only, provided via sample_id. Distances reflect a single baseline model.'
    }
    (target/'contact_sheet_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


def scatter_plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    df=read_csv(args.source_zip,'feature_confusion_join.csv')
    required={'run_id','class_i_name','class_j_name','centroid_cosine_similarity','symmetric_confusion_rate_pct'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    target=args.output_dir.resolve();target.mkdir(parents=True,exist_ok=True)
    generated=[]
    for run_id in MISSING_SCATTERS:
        sub=df[df.run_id==run_id].copy()
        if len(sub)!=435:
            raise ValueError(f'Expected 435 unordered pairs for {run_id}, got {len(sub)}')
        x=sub.centroid_cosine_similarity.astype(float)
        y=sub.symmetric_confusion_rate_pct.astype(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('NaN/Inf in input plot data')
        rho=x.corr(y,method='spearman')
        flag=sub.apply(lambda r:frozenset((r.class_i_name,r.class_j_name)) in LITERATURE_PAIRS,axis=1)
        fig,ax=plt.subplots(figsize=(9,6.4))
        ax.scatter(x[~flag],y[~flag],s=17,alpha=.35,label='All other class pairs')
        ax.scatter(x[flag],y[flag],s=55,alpha=.95,marker='D',label='Literature-selected pairs')
        # Highlight selected pairs without overlapping text annotations.
        ax.set_xlabel('Cosine similarity between class centroids (unitless)')
        ax.set_ylabel('Symmetric mutual confusion rate (%)')
        ax.set_title(f'Similarity vs confusion - {run_id}\nSpearman rho = {rho:+.3f}; 435 class pairs')
        ax.grid(alpha=.2)
        ax.legend(loc='upper left',fontsize=9)
        fig.tight_layout()
        fn=f'similarity_vs_confusion_{run_id}.png'
        fig.savefig(target/fn,dpi=300,bbox_inches='tight')
        plt.close(fig)
        generated.append({'run_id':run_id,'figure':fn,'n_pairs':len(sub),'spearman_rho':round(float(rho),6)})
    pdfile=target/'missing_scatter_inventory.csv'
    with pdfile.open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(generated[0]));writer.writeheader();writer.writerows(generated)
    print(json.dumps({'status':'passed','generated_figures':generated,'inventory':str(pdfile)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action',choices=('contact','scatters'))
    p.add_argument('--source-zip',type=Path,required=True,help='ai_ready_evaluation_with_feature_space.zip')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/feature_analysis/figures'))
    p.add_argument('--aid-root',type=Path,help='Original AID root containing class folders (contact only)')
    p.add_argument('--run-id',default=DEFAULT_RUN,help='One baseline model for contact sheets')
    p.add_argument('--selection-csv',type=Path,default=None,help='Optional original CSV instead of ZIP copy')
    a=p.parse_args()
    if not a.source_zip.is_file():
        p.error(f'ZIP does not exist: {a.source_zip}')
    try:
        if a.action=='contact': contact_sheet(a)
        else: scatter_plots(a)
    except Exception as e:
        print(json.dumps({'status':'failed','error':f'{type(e).__name__}: {e}'},ensure_ascii=False,indent=2),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
