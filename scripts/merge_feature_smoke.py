#!/usr/bin/env python3
"""Merge PC and server smoke audits. Require one passed checkpoint per EFCIL method.

Usage: python scripts/merge_feature_smoke.py --audits <pc.csv> <server.csv> \
       --output outputs/feature_analysis/global_smoke_summary.json
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
METHODS = ('fetril','fecam','ssre','pass','efc')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audits', nargs='+', required=True)
    p.add_argument('--output', default='outputs/feature_analysis/global_smoke_summary.json')
    a=p.parse_args()
    rows=[]
    for f in a.audits:
        with Path(f).open(encoding='utf-8-sig',newline='') as fp:
            rows += list(csv.DictReader(fp))
    passed={m: [r for r in rows if r.get('method')==m and r.get('status')=='passed' and
                str(r.get('checkpoint_loadable','')).lower()=='true' and
                str(r.get('feature_extractable','')).lower()=='true' and
                str(r.get('y_true_match','')).lower()=='true' and
                str(r.get('feature_dim',''))=='512'] for m in METHODS}
    good=all(passed[m] for m in METHODS)
    data={"generated_at":datetime.now(timezone.utc).isoformat(), "ready_for_batch":good,
          "per_method": {m: [{k:r.get(k) for k in ('run_id','machine','checkpoint_path','feature_dim','sample_identity_match')} for r in passed[m]] for m in METHODS},
          "requirements": "One real production final-task checkpoint per method loadable + 512-d feature + canonical y_true match"}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Global five-method loadability smoke:', 'PASSED' if good else 'BLOCKED')
    for m in METHODS: print(m, 'passed' if passed[m] else 'MISSING')
    print('Saved:',out)
    if not good: raise SystemExit(2)

if __name__=='__main__': main()
