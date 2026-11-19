#!/usr/bin/env python3
"""Consolidate and verify 80-run production feature extraction audits."""
import argparse
import json
from pathlib import Path
import pandas as pd
import zipfile
import io


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pc-audit',type=Path,required=True)
    ap.add_argument('--server-audit',type=Path,required=True)
    ap.add_argument('--source-zip',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/feature_analysis/audit_consolidated'))
    args=ap.parse_args()
    pc=pd.read_csv(args.pc_audit).assign(_source='pc_audit')
    sv=pd.read_csv(args.server_audit).assign(_source='server_audit')
    x=pd.concat([pc,sv],ignore_index=True)
    with zipfile.ZipFile(args.source_zip) as z:
        reference=pd.read_csv(io.BytesIO(z.read('model_insights_all.csv')))
        pairs=pd.read_csv(io.BytesIO(z.read('attention_insights_all.csv')))
    expected=set(reference.run_id.dropna().astype(str)) if 'run_id' in reference else set()
    if len(expected)!=80:
        raise ValueError(f'Expected 80 ready production model run IDs, found {len(expected)}')
    duplicate=x[x.run_id.duplicated(keep=False)].run_id.tolist()
    errors=[]
    if duplicate: errors.append(f'Duplicate run IDs: {duplicate}')
    if len(x)!=80:errors.append(f'Expected exactly 80 audit rows, got {len(x)}')
    got=set(x.run_id.astype(str))
    if missing:=sorted(expected-got):errors.append(f'Missing: {missing}')
    if extra:=sorted(got-expected):errors.append(f'Unexpected: {extra}')
    for col in ('checkpoint_loadable','feature_extractable','y_true_match'):
        if not x[col].fillna(False).astype(bool).all():errors.append(f'{col} false on {int((~x[col].fillna(False).astype(bool)).sum())} rows')
    if not x.status.eq('passed').all():errors.append('Not all status=passed')
    if not x.feature_dim.eq(512).all():errors.append('Feature dim mismatch')
    if not x.n_samples.eq(2000).all():errors.append('Sample count mismatch')
    if not x.seed.eq(1993).all():errors.append('Seed mismatch')
    if not ((x.machine.eq('pc2') & x._source.eq('pc_audit')) | (x.machine.eq('server') & x._source.eq('server_audit'))).all():errors.append('Machine/source mismatch')
    x['checkpoint_relative_path']=x.apply(lambda r:f"{r.run_id}/model_checkpoints/model_{4 if r.protocol=='T5' else 9}.pkl",axis=1)
    x=x.drop(columns=['checkpoint_path','_source'])
    keys=['method','protocol','backbone','seed','batch','machine']
    baseline=x[x.attention=='none'].set_index(['method','protocol','backbone','seed'])
    comparison=x[x.attention!='none']
    n_fair=0
    for r in comparison.itertuples(index=False):
        key=(r.method,r.protocol,r.backbone,r.seed)
        if key not in baseline.index:
            errors.append(f'No baseline for {r.run_id}');continue
        b=baseline.loc[key]
        if r.batch!=b.batch or r.machine!=b.machine:
            errors.append(f'Unfair pair for {r.run_id}');continue
        n_fair+=1
    if n_fair!=60:errors.append(f'Expected 60 feature fairness keys matched, got {n_fair}')
    if 'status' in pairs.columns:
        n_paired=(pairs.status=='paired_valid').sum()
        if n_paired!=60:errors.append(f'AI ready main evaluation only {n_paired}/60 valid pairs')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    x=x.sort_values(['method','protocol','backbone','attention']).reset_index(drop=True)
    (out/'feature_extraction_audit_consolidated_80.csv').write_bytes(x.to_csv(index=False).encode('utf-8-sig'))
    status={
        'status':'passed' if not errors else 'failed',
        'n_runs':int(len(x)),
        'n_pc2':int((x.machine=='pc2').sum()),
        'n_server':int((x.machine=='server').sum()),
        'n_methods':int(x.method.nunique()),
        'n_valid_feature_pairs':n_fair,
        'n_embedding_vectors':int(x.n_samples.sum()),
        'feature_dim_unique':sorted(int(i) for i in x.feature_dim.unique()),
        'all_sample_identity_match': sorted(x.sample_identity_match.dropna().unique().tolist()),
        'caveat':'label_order_only does not establish exact image identity from canonical prediction artifacts.',
        'errors':errors,
    }
    (out/'feature_extraction_consolidated_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
