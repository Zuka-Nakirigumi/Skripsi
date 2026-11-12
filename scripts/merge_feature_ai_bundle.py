#!/usr/bin/env python3
"""Add feature CSV/JSON to a COPY of the existing audited AI-ready ZIP.

Never alters original 08 exporter outputs or includes checkpoint weights/datasets/embeddings.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
FILES = [
    'feature_analysis_manifest.json', 'feature_analysis_data_dictionary.json',
    'feature_analysis_insights.json', 'feature_analysis_insights_all.json',
    'prediction_npz_audit.csv', 'feature_checkpoint_audit.csv',
    'feature_extraction_audit.csv', 'feature_analysis_errors.csv',
    'feature_analysis_validated_runs.csv',
    'class_feature_statistics.csv', 'inter_class_similarity.csv',
    'inter_class_similarity_top_pairs.csv', 'intra_class_dispersion.csv',
    'feature_confusion_join.csv', 'feature_correlation_summary.csv',
    'attention_feature_delta.csv', 'attention_feature_summary.csv',
    'literature_class_pairs_analysis.csv',
    'table_inter_class_literature_pairs.csv', 'table_intra_class_top.csv',
    'table_attention_feature_summary.csv', 'feature_contact_sheet_selection.csv',
]

def merge(ai_zip, feature_dir, output):
    ai_zip=Path(ai_zip); feature_dir=Path(feature_dir);output=Path(output)
    if not ai_zip.is_file(): raise FileNotFoundError(ai_zip)
    if not (feature_dir/'feature_analysis_manifest.json').is_file():
        raise FileNotFoundError('Missing feature_analysis_manifest.json; run feature analysis first')
    # Prevent combining unrelated/stale analysis outputs with a different AI-ready run grid.
    manifest=json.loads((feature_dir/'feature_analysis_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(ai_zip) as check:
        if 'model_insights_all.csv' not in check.namelist():
            raise RuntimeError('AI ZIP missing model_insights_all.csv; use ALL MODELS v2.1 exporter')
        runs=pd.read_csv(io.BytesIO(check.read('model_insights_all.csv')))
        ai_ids=set(runs.run_id.dropna().astype(str))
        feature_ids=set(manifest.get('extracted_run_ids',[]))
        invalid=feature_ids-ai_ids
        if invalid:
            raise RuntimeError(f'Feature evidence includes run_id absent from AI-ready ZIP: {sorted(invalid)[:5]}')
        if str(manifest.get('dataset')).lower()!='aid' or int(manifest.get('seed',-1))!=1993:
            raise RuntimeError('Unexpected feature dataset/seed; refuse merge')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(ai_zip) as zi, zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as zo:
        for item in zi.infolist():
            if item.filename in FILES: continue
            zo.writestr(item,zi.read(item.filename))
        for f in FILES:
            p=feature_dir/f
            if p.is_file(): zo.write(p,arcname='feature_space/'+f)
    tmp.replace(output)
    with zipfile.ZipFile(output) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'ZIP CRC failed: {bad}')
        print('Written:',output, '| files:',len(z.namelist()),'| feature files:',len([x for x in z.namelist() if x.startswith('feature_space/')]))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ai-zip',required=True)
    p.add_argument('--feature-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    merge(a.ai_zip,a.feature_dir,a.output)
