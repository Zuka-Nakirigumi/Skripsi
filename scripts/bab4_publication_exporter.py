"""Publication-ready Bab IV (AID), powered by audited AI-ready feature-space ZIP.

Self-contained, offline. No training, checkpoint loading, or recalculation of
canonical AIA/LTA/FR/BWT. The original notebook can also embed this source.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

METHODS_PUB = ('fetril', 'fecam', 'ssre', 'pass', 'efc')
BACKBONES_PUB = ('r18', 'r34')
PROTOCOLS_PUB = ('T5', 'T10')
ATTNS_PUB = ('eca', 'cbam', 'procbam')
METHOD_NAMES_PUB = {'fetril': 'FeTrIL', 'fecam': 'FeCAM', 'ssre': 'SSRE', 'pass': 'PASS', 'efc': 'EFC'}
LITERATURE_PAIRS_PUB = (
    ('Stadium', 'Playground'), ('BareLand', 'Desert'), ('Resort', 'Park'),
    ('Church', 'Commercial'), ('Beach', 'Port'), ('Square', 'Center'),
    ('MediumResidential', 'DenseResidential'),
)
BUNDLE_REQUIRED_PUB = {
    'model_insights_all.csv', 'paired_attention_effects.csv', 'attention_insights_all.csv',
    'class_insights_final_all.csv', 'class_accuracy_long.csv', 'misclassification_long.csv',
    'class_label_map.csv', 'task_attention_delta.csv', 'class_forgetting_attention_delta.csv',
    'misclassification_delta.csv',
    'feature_space/feature_analysis_manifest.json',
    'feature_space/inter_class_similarity.csv', 'feature_space/feature_confusion_join.csv',
    'feature_space/intra_class_dispersion.csv', 'feature_space/attention_feature_summary.csv',
    'feature_space/feature_contact_sheet_selection.csv',
}


def _assert_columns(df, name, columns):
    absent = set(columns) - set(df.columns)
    if absent:
        raise ValueError(f'{name} tidak memiliki kolom wajib: {sorted(absent)}')


def _write_table(df, target, caption):
    df.to_csv(target, index=False, encoding='utf-8-sig', float_format='%.7f')
    return {'name': target.name, 'caption': caption, 'rows': len(df), 'columns': len(df.columns), 'status': 'generated'}


def _pair_sort(df):
    d = df.copy()
    d['_method_order'] = pd.Categorical(d['method'], categories=METHODS_PUB, ordered=True)
    d['_backbone_order'] = pd.Categorical(d['backbone'], categories=BACKBONES_PUB, ordered=True)
    d['_protocol_order'] = pd.Categorical(d['protocol'], categories=PROTOCOLS_PUB, ordered=True)
    d['_attention_order'] = pd.Categorical(d['attention'], categories=ATTNS_PUB, ordered=True)
    return d.sort_values(['_method_order', '_backbone_order', '_protocol_order', '_attention_order']).drop(columns=['_method_order','_backbone_order','_protocol_order','_attention_order'])


def load_bab4_evidence(bundle_path):
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f'ZIP gabungan evaluasi tidak ditemukan: {bundle_path}')
    result = {}
    with zipfile.ZipFile(bundle_path) as zin:
        if zin.testzip() is not None:
            raise ValueError('File ZIP gagal pemeriksaan CRC.')
        missing = BUNDLE_REQUIRED_PUB - set(zin.namelist())
        if missing:
            raise FileNotFoundError(f'File evidence wajib hilang dari ZIP: {sorted(missing)}')
        for name in zin.namelist():
            if name.endswith('.csv'):
                if name.endswith('feature_analysis_errors.csv') and zin.getinfo(name).file_size <= 5:
                    continue
                try:
                    result[name] = pd.read_csv(io.BytesIO(zin.read(name)))
                except pd.errors.EmptyDataError:
                    result[name] = pd.DataFrame()
            elif name.endswith('.json'):
                result[name] = json.loads(zin.read(name).decode('utf-8-sig'))
    runs = result['model_insights_all.csv']
    pairs = result['paired_attention_effects.csv']
    fs = result['feature_space/attention_feature_summary.csv']
    _assert_columns(runs, 'model_insights_all.csv', ['dataset','method','protocol','backbone','attention','seed','aia_pct','lta_pct','run_id'])
    _assert_columns(pairs, 'paired_attention_effects.csv', ['method','protocol','backbone','attention','delta_aia_pp','delta_lta_pp','delta_fr_pp','delta_bwt_pp','baseline_run_id','attention_run_id'])
    _assert_columns(fs, 'attention_feature_summary.csv', ['baseline_run_id','attention_run_id','mean_delta_inter_class_similarity','mean_delta_intra_dispersion','mean_delta_mutual_confusion_pp','delta_aia_pp','delta_lta_pp'])
    if len(runs) != 80 or len(pairs) != 60 or len(fs) != 60:
        raise ValueError(f'Coverage tidak lengkap: model={len(runs)}/80, pairs={len(pairs)}/60, feature summary={len(fs)}/60. Tidak membuat tabel final dari data parsial.')
    expected_keys = set(product(METHODS_PUB, PROTOCOLS_PUB, BACKBONES_PUB, ('none',) + ATTNS_PUB))
    actual_keys = set(map(tuple, runs[['method','protocol','backbone','attention']].itertuples(index=False, name=None)))
    if expected_keys != actual_keys:
        raise ValueError(f'Grid model tidak tepat: missing={len(expected_keys-actual_keys)}, extra={len(actual_keys-expected_keys)}')
    if not runs['ready'].fillna(False).all():
        raise ValueError('Tidak semua 80 model berstatus ready.')
    if runs['run_id'].duplicated().any() or pairs.duplicated(['method','protocol','backbone','attention']).any():
        raise ValueError('Ditemukan duplikat konfigurasi/run_id pada ZIP.')
    assert len(result['feature_space/inter_class_similarity.csv']) == 80 * 435
    assert len(result['feature_space/intra_class_dispersion.csv']) == 80 * 30
    assert len(result['feature_space/feature_confusion_join.csv']) == 80 * 435
    assert len(result['class_insights_final_all.csv']) == 60 * 30
    fv = result['feature_space/feature_analysis_validated_runs.csv']
    _assert_columns(fv, 'feature_analysis_validated_runs.csv', ['run_id','feature_dim','n_samples','status'])
    if (len(fv)!=80 or fv.run_id.duplicated().any() or set(fv.run_id)!=set(runs.run_id)
        or not (fv.feature_dim==512).all() or not (fv.n_samples==2000).all()
        or not (fv.status=='validated').all()):
        raise ValueError('Audit feature-space 80 run tidak cocok dengan model utama.')
    # Verify exact joins, and that delta AIA/LTA in feature summaries are the canonical deltas.
    merged=fs.merge(pairs[['attention_run_id','baseline_run_id','delta_aia_pp','delta_lta_pp','source_machine','batch']],
                    on=['attention_run_id','baseline_run_id'],how='outer',validate='one_to_one',suffixes=('_feature','_canonical'),indicator=True)
    if len(merged)!=60 or not (merged['_merge']=='both').all():
        raise ValueError('Ada pasangan feature-space yang tidak ada pada paired_attention_effects.csv.')
    for m in ('delta_aia_pp','delta_lta_pp'):
        if not np.allclose(merged[m+'_feature'],merged[m+'_canonical'],atol=1e-7,rtol=0):
            raise ValueError(f'Metrik canonical {m} tidak cocok dengan feature summary.')
    if not (merged.batch_feature==merged.batch_canonical).all() or not (merged.source_machine_feature==merged.source_machine_canonical).all():
        raise ValueError('Fairness batch/machine pada pasangan feature tidak cocok dengan canonical pairs.')
    manifest=result['feature_space/feature_analysis_manifest.json']
    if int(manifest.get('validated_embedded_runs', 80))!=80 or int(manifest.get('n_fair_feature_attention_pairs',60))!=60:
        raise ValueError('Manifest feature-space menyatakan coverage berbeda dari 80 model / 60 pairs.')
    return result


def build_tables_bab4(e, table_dir):
    table_dir = Path(table_dir); table_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    runs = e['model_insights_all.csv'].copy()
    pairs = e['paired_attention_effects.csv'].copy()
    feature = e['feature_space/attention_feature_summary.csv'].copy()
    class_final = e['class_insights_final_all.csv'].copy()
    keyidx = {tuple(r[k] for k in ['method','backbone','protocol','attention']): r for _,r in runs.iterrows()}
    pidx = {tuple(r[k] for k in ['method','backbone','protocol','attention']): r for _,r in pairs.iterrows()}

    # 4.1-4.4: Canonical run metrics, no recomputation.
    for no,protocol,metric,caption in [
        (1,'T5','aia_pct','AIA T=5 (%)'), (2,'T5','lta_pct','LTA T=5 (%)'),
        (3,'T10','aia_pct','AIA T=10 (%)'), (4,'T10','lta_pct','LTA T=10 (%)'),
    ]:
        rows = []
        for method,backbone in product(METHODS_PUB,BACKBONES_PUB):
            r={'Method':METHOD_NAMES_PUB[method], 'Backbone':f'ResNet-{backbone[1:]}'}
            for att in ('none',)+ATTNS_PUB:
                r[{'none':'None','eca':'ECA','cbam':'CBAM','procbam':'ProCBAM'}[att]] = float(keyidx[method,backbone,protocol,att][metric])
            rows.append(r)
        recs.append(_write_table(pd.DataFrame(rows),table_dir/f'Tabel_4_{no:02d}.csv',caption))

    # 4.5-4.6: Six paired deltas / method-backbone row.
    for no,protocol in [(5,'T5'),(6,'T10')]:
        rows=[]
        for method,backbone in product(METHODS_PUB,BACKBONES_PUB):
            r={'Method':METHOD_NAMES_PUB[method], 'Backbone':f'ResNet-{backbone[1:]}'}
            for att in ATTNS_PUB:
                x=pidx[method,backbone,protocol,att]
                r[f'ΔAIA {att.upper()} (pp)']=float(x.delta_aia_pp)
                r[f'ΔLTA {att.upper()} (pp)']=float(x.delta_lta_pp)
            rows.append(r)
        recs.append(_write_table(pd.DataFrame(rows),table_dir/f'Tabel_4_{no:02d}.csv',f'Delta AIA/LTA T={protocol[1:]} (pp)'))

    # 4.7 FR/BWT: lower FR is better, higher BWT is better.
    rows=[]
    for a in ATTNS_PUB:
        g=pairs[pairs.attention==a]
        rows.append({'Attention':a.upper(),'N pairs':len(g),'FR membaik (ΔFR<0)':int((g.delta_fr_pp<0).sum()),
                     'Mean ΔFR (pp)':g.delta_fr_pp.mean(), 'BWT membaik (ΔBWT>0)':int((g.delta_bwt_pp>0).sum()),
                     'Mean ΔBWT (pp)':g.delta_bwt_pp.mean()})
    recs.append(_write_table(pd.DataFrame(rows),table_dir/'Tabel_4_07.csv','Ringkasan FR dan BWT'))

    # 4.8 balanced extremes per-class forgetting; negative delta means improvement.
    forg = e['class_forgetting_attention_delta.csv'].copy()
    forg = forg[forg.support>=10].copy()
    fsort = ['delta_forgetting_pp','method','protocol','backbone','attention','class_id']
    fg = forg[forg.delta_forgetting_pp < 0].sort_values(fsort).head(4).copy()
    fb = forg[forg.delta_forgetting_pp > 0].sort_values(fsort, ascending=[False,True,True,True,True,True]).head(4).copy()
    f8=pd.concat([fg,fb],ignore_index=True)
    f8.insert(0,'Outcome', ['Forgetting berkurang']*len(fg)+['Forgetting meningkat']*len(fb))
    f8=f8.rename(columns={'class_name':'Class','baseline_forgetting_pp':'FR None (pp)',
                          'attention_forgetting_pp':'FR Attention (pp)','delta_forgetting_pp':'ΔForgetting (pp)'})
    recs.append(_write_table(f8[['Outcome','method','protocol','backbone','attention','Class','FR None (pp)','FR Attention (pp)','ΔForgetting (pp)','support','baseline_run_id','attention_run_id']],table_dir/'Tabel_4_08.csv','Contoh per-class forgetting terbesar (4 membaik, 4 memburuk)'))

    # 4.9: macro-average over conditions, not raw task-weighted mean; exclude task0.
    td = e['task_attention_delta.csv'].copy(); td=td[td.task_id>0].copy()
    condition_cols=['dataset','method','protocol','backbone','seed','attention']
    cond=(td.groupby(condition_cols,as_index=False)
          .agg(mean_delta_old_pp=('delta_old_pp','mean'),mean_delta_new_pp=('delta_new_pp','mean')))
    rows=[]
    for att in ATTNS_PUB:
        g=cond[cond.attention==att]; task=td[td.attention==att]
        rows.append({'Attention':att.upper(),'N paired conditions':len(g),
                     'Mean ΔOld Accuracy (pp)':g.mean_delta_old_pp.mean(),
                     'Mean ΔNew Accuracy (pp)':g.mean_delta_new_pp.mean(),
                     'Old tasks meningkat':int((task.delta_old_pp>0).sum()),
                     'New tasks meningkat':int((task.delta_new_pp>0).sum()),
                     'N incremental tasks evaluated':len(task)})
    t9=pd.DataFrame(rows)
    recs.append(_write_table(t9,table_dir/'Tabel_4_09.csv','Ringkasan old/new accuracy, macro mean per 20 pasangan'))

    # 4.10 Macro-F1 and class changes on final task (60×30 class rows).
    t10=[]
    for att in ATTNS_PUB:
        g=pairs[pairs.attention==att]
        c=class_final[class_final.attention==att]
        t10.append({'Attention':att.upper(),'Macro-F1 naik (pairs)':int((g.delta_macro_f1_pp>0).sum()),
                    'N pairs':len(g),'Mean ΔMacro-F1 (pp)':g.delta_macro_f1_pp.mean(),
                    'Class accuracy naik':int((c.delta_class_accuracy_pp>1e-9).sum()),
                    'Class accuracy turun':int((c.delta_class_accuracy_pp< -1e-9).sum()),
                    'Class accuracy tetap':int((c.delta_class_accuracy_pp.abs()<=1e-9).sum()),
                    'N class observations':len(c),'Mean ΔClass Accuracy (pp)':c.delta_class_accuracy_pp.mean()})
    recs.append(_write_table(pd.DataFrame(t10),table_dir/'Tabel_4_10.csv','Macro-F1 dan class accuracy final'))

    # 4.11 extremes with deterministic tie-break and preserved source run ids.
    base_keys=['delta_class_accuracy_pp','method','protocol','backbone','attention','class_id']
    cp=class_final[class_final.delta_class_accuracy_pp>0].sort_values(base_keys,ascending=[False,True,True,True,True,True]).head(4)
    cn=class_final[class_final.delta_class_accuracy_pp<0].sort_values(base_keys).head(4)
    ex=pd.concat([cp,cn],ignore_index=True)
    ex.insert(0,'Outcome',['Kenaikan']*len(cp)+['Penurunan']*len(cn))
    ex=ex.rename(columns={'class_name':'Class','baseline_class_accuracy_pct':'Baseline accuracy (%)',
                          'attention_class_accuracy_pct':'Attention accuracy (%)',
                          'delta_class_accuracy_pp':'ΔAccuracy (pp)','attention_support':'Support'})
    ex_cols=['Outcome','method','protocol','backbone','attention','Class','Support','Baseline accuracy (%)','Attention accuracy (%)','ΔAccuracy (pp)','baseline_run_id','attention_run_id']
    recs.append(_write_table(ex[ex_cols],table_dir/'Tabel_4_11.csv','Extreme final class accuracy: empat naik, empat turun'))

    # 4.12 same unordered literature pairs, aggregate across 20 baseline representations.
    lc=e['feature_space/feature_confusion_join.csv']
    lc=lc[lc.attention=='none'].copy()
    lc['pair_key']=lc.apply(lambda r:frozenset((str(r.class_i_name).casefold(),str(r.class_j_name).casefold())),axis=1)
    t12=[]
    for first,second in LITERATURE_PAIRS_PUB:
        g=lc[lc.pair_key==frozenset((first.casefold(),second.casefold()))]
        if len(g)!=20:
            raise ValueError(f'Pasangan literatur {first}↔{second} memiliki {len(g)} baseline, diharapkan 20')
        t12.append({'Pair':f'{first} ↔ {second}', 'N baseline models':len(g),
                    'Mean centroid cosine similarity':g.centroid_cosine_similarity.mean(),
                    'Mean symmetric confusion (%)':g.symmetric_confusion_rate_pct.mean(),
                    'Min similarity':g.centroid_cosine_similarity.min(),
                    'Max similarity':g.centroid_cosine_similarity.max()})
    recs.append(_write_table(pd.DataFrame(t12),table_dir/'Tabel_4_12.csv','Pasangan kelas literatur: mean over 20 baseline models'))

    # 4.13 baseline-only aggregation avoids mixing attention with baseline.
    intra=e['feature_space/intra_class_dispersion.csv']
    intra=intra[intra.attention=='none'].copy()
    t13=(intra.groupby(['class_id','class_name'],as_index=False)
         .agg(n_baseline_models=('run_id','nunique'),
              mean_dispersion=('mean_cosine_distance_to_centroid','mean'),
              mean_final_error_pct=('final_class_error_rate_pct','mean'),
              mean_forgetting_pp=('class_forgetting_pp','mean'),
              mean_support=('support','mean')))
    if len(t13)!=30 or (t13.n_baseline_models!=20).any():
        raise ValueError('Tabel 4.13 memerlukan 30 kelas × 20 model baseline lengkap')
    t13=t13.sort_values(['mean_dispersion','class_id'],ascending=[False,True]).reset_index(drop=True)
    t13.insert(0,'Rank',np.arange(1,len(t13)+1))
    recs.append(_write_table(t13.head(10),table_dir/'Tabel_4_13.csv','Sepuluh kelas baseline dengan mean feature dispersion tertinggi'))
    _write_table(t13,table_dir/'Lampiran_dispersion_30_kelas.csv','Ranking 30 kelas baseline')

    # 4.14 summary across 20 valid conditions for each attention.
    fs_rows=[]
    for att in ATTNS_PUB:
        g=feature[feature.attention==att]
        if len(g)!=20:
            raise ValueError('Tabel 4.14 membutuhkan 20 pasangan / attention')
        fs_rows.append({'Attention':att.upper(),'N fair pairs':len(g),
                        'Mean ΔAIA (pp)':g.delta_aia_pp.mean(),'Mean ΔLTA (pp)':g.delta_lta_pp.mean(),
                        'Mean ΔCentroid Similarity':g.mean_delta_inter_class_similarity.mean(),
                        'Similarity turun (pairs)':int((g.mean_delta_inter_class_similarity<0).sum()),
                        'Mean ΔIntra Dispersion':g.mean_delta_intra_dispersion.mean(),
                        'Dispersion turun (pairs)':int((g.mean_delta_intra_dispersion<0).sum()),
                        'Mean ΔMutual Confusion (pp)':g.mean_delta_mutual_confusion_pp.mean(),
                        'Confusion turun (pairs)':int((g.mean_delta_mutual_confusion_pp<0).sum())})
    t14=pd.DataFrame(fs_rows)
    recs.append(_write_table(t14,table_dir/'Tabel_4_14.csv','Ringkasan efek attention dalam feature-space'))

    # 4.15 derive representative cases if feature_representative_cases.csv was NOT bundled.
    selectors=[
        ('Largest mean confusion reduction','mean_delta_mutual_confusion_pp','min'),
        ('Largest mean centroid similarity reduction','mean_delta_inter_class_similarity','min'),
        ('Largest mean compactness improvement','mean_delta_intra_dispersion','min'),
        ('Largest AIA degradation','delta_aia_pp','min'),
    ]
    cases=[]
    for label,col,direction in selectors:
        g=feature.sort_values([col,'method','protocol','backbone','attention'],ascending=[direction!='max',True,True,True,True])
        # Only mark degradation if actual AIA is negative.
        if label.endswith('degradation'):
            g=g[g.delta_aia_pp<0]
        if g.empty:continue
        x=g.iloc[0]
        cases.append({'Criterion':label,'Selected metric':col,'Metric value':float(x[col]),
                      'Method':METHOD_NAMES_PUB[x.method],'Protocol':x.protocol,
                      'Backbone':x.backbone.upper(),'Attention':x.attention.upper(),
                      'ΔAIA (pp)':float(x.delta_aia_pp), 'ΔLTA (pp)':float(x.delta_lta_pp),
                      'Mean ΔSimilarity':float(x.mean_delta_inter_class_similarity),
                      'Mean ΔDispersion':float(x.mean_delta_intra_dispersion),
                      'Mean ΔMutual Confusion (pp)':float(x.mean_delta_mutual_confusion_pp),
                      'Baseline Run ID':x.baseline_run_id,'Attention Run ID':x.attention_run_id})
    recs.append(_write_table(pd.DataFrame(cases),table_dir/'Tabel_4_15.csv','Kasus representatif berdasarkan kriteria objektif'))

    # 4.16 off-diagonal errors, balanced across improvements and degradations, with predefined cases.
    md=e['misclassification_delta.csv']
    positives=md[md.errors_reduced>0].sort_values(['errors_reduced','method','protocol','backbone','attention','true_class_id','predicted_class_id'],ascending=[False,True,True,True,True,True,True]).head(5)
    negatives=md[md.errors_reduced<0].sort_values(['errors_reduced','method','protocol','backbone','attention','true_class_id','predicted_class_id']).head(5)
    curated=md[((md.method=='fecam')&(md.protocol=='T10')&(md.backbone=='r18')&(md.attention=='cbam')&
                (md.true_class=='Desert')&(md.predicted_class=='BareLand')) |
               ((md.method=='pass')&(md.protocol=='T10')&(md.backbone=='r18')&(md.attention=='cbam')&
                (md.true_class=='Pond')&(md.predicted_class=='Park'))]
    t16=pd.concat([curated,positives,negatives],ignore_index=True).drop_duplicates(subset=['baseline_run_id','attention_run_id','true_class_id','predicted_class_id'])
    t16=t16.copy();t16.insert(0,'Interpretation',np.where(t16.errors_reduced>0,'Errors berkurang',np.where(t16.errors_reduced<0,'Errors meningkat','Tidak berubah')))
    t16=t16.rename(columns={'true_class':'True class','predicted_class':'Predicted class','errors_none':'Errors None',
                             'errors_attention':'Errors Attention','errors_reduced':'Errors reduced (count)','delta_error_rate_pp':'ΔError Rate (pp)'})
    recs.append(_write_table(t16[['Interpretation','method','protocol','backbone','attention','True class','Predicted class',
                                  'support','Errors None','Errors Attention','Errors reduced (count)','ΔError Rate (pp)','baseline_run_id','attention_run_id']],
                            table_dir/'Tabel_4_16.csv','Misclassification changes: peningkatan dan degradasi'))
    return recs, {'table_4_9':t9, 'table_4_11':ex,'table_4_14':t14,'table_4_15':pd.DataFrame(cases)}


def _plot_heatmap(values, row_labels, col_labels, title, dest, reverse=False, value_format='{:+.2f}', x_label='Attention mechanism', y_label='Method / Backbone / Protocol'):
    a=np.asarray(values,dtype=float)
    max_abs=max(0.1,np.nanmax(np.abs(a)))
    fig,ax=plt.subplots(figsize=(9.5,10),constrained_layout=True)
    im=ax.imshow(a,cmap='RdYlGn_r' if reverse else 'RdYlGn',norm=TwoSlopeNorm(vmin=-max_abs,vcenter=0,vmax=max_abs),aspect='auto')
    ax.set_xticks(range(len(col_labels)),col_labels,fontsize=10)
    ax.set_yticks(range(len(row_labels)),row_labels,fontsize=8)
    ax.set_title(title,fontsize=12,pad=14)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i,j]):
                ax.text(j,i,value_format.format(a[i,j]),ha='center',va='center',fontsize=7,color='black')
    fig.colorbar(im,ax=ax,label='Δ metric (percentage points)',fraction=0.044,pad=0.03)
    fig.savefig(dest,dpi=300,bbox_inches='tight');plt.close(fig)


def _label_map(e):
    names=e['class_label_map.csv'].sort_values('incremental_id')
    if len(names)!=30 or names['incremental_id'].tolist()!=list(range(30)):
        raise ValueError('class_label_map AID harus 30 ID incremental lengkap (0..29)')
    if not names['class_label_verified'].astype(bool).all():
        raise ValueError('Nama kelas AID belum terverifikasi; jangan gunakan label Cxx sebagai nama scene.')
    return names.class_name.astype(str).tolist()


def _build_cm_from_csv(e, run_id, raw_sources=None):
    """Reconstruct exact 30×30 counts from per-class TP + all nonzero off-diagonal errors.

    Verifies each row support and total against canonical last task accuracy.
    Never interpolates or fabricates matrix elements.
    """
    final=e['class_accuracy_long.csv']; mis=e['misclassification_long.csv']
    run=e['model_insights_all.csv']; match=run[run.run_id==run_id]
    if len(match)!=1:raise ValueError(f'Run tidak tunggal: {run_id}')
    r=match.iloc[0];task_id=4 if r.protocol=='T5' else 9
    g=final[(final.run_id==run_id)&(final.task_id==task_id)].sort_values('class_id')
    if len(g)!=30 or g.class_id.tolist()!=list(range(30)):
        raise ValueError(f'Class accuracy final tidak lengkap: {run_id}')
    cm=np.zeros((30,30),dtype=np.int64)
    for row in g.itertuples(index=False):
        cm[int(row.class_id),int(row.class_id)]=int(row.correct)
    off=mis[mis.run_id==run_id]
    for row in off.itertuples(index=False):
        i,j=int(row.true_class_id),int(row.predicted_class_id)
        if i==j or cm[i,j]:raise ValueError(f'Duplicate/off-diagonal invalid untuk {run_id}: {i},{j}')
        cm[i,j]=int(row.errors)
    if not np.array_equal(cm.sum(axis=1),g.support.to_numpy(dtype=int)):
        raise ValueError(f'Jumlah confusion matrix per kelas tidak sesuai support: {run_id}')
    if int(cm.sum())!=2000:
        raise ValueError(f'AID final test support harus 2000, didapat {cm.sum()} untuk {run_id}')
    top1=float(np.trace(cm)/cm.sum()*100)
    if abs(top1-float(r.lta_pct))>0.055:
        raise ValueError(f'Rekonstruksi confusion tidak cocok LTA: {run_id} {top1:.5f} vs {r.lta_pct:.5f}')
    # If raw npy files exist, assert counts agree; reconstruction remains portable.
    if raw_sources:
        for root in raw_sources:
            path=Path(root)/run_id/'final_confusion_matrix.npy'
            if path.is_file():
                original=np.load(path,allow_pickle=False)
                if not np.array_equal(original,cm):
                    raise ValueError(f'Reconstructed confusion berbeda dari final_confusion_matrix.npy: {run_id}')
                break
    return cm


def _safe_contact_sheet_from_metadata(e, output_path, root, run_id, kind):
    """Strict real-image renderer. Missing image raises; never uses placeholders."""
    from PIL import Image, ImageDraw, ImageOps
    root=Path(root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(f'Folder AID belum tersedia: {root}')
    samples=e['feature_space/feature_contact_sheet_selection.csv']
    sub=samples[samples.run_id==run_id].copy()
    if len(sub)<150:raise ValueError(f'Tidak ada pemilihan sampel lengkap untuk {run_id}')
    selects=['closest','median','p90','farthest']
    if kind=='inter':
        class_rows=[cl for pair in [('Desert','BareLand'),('Stadium','Playground'),('Resort','Park')] for cl in pair]
        title='Inter-class examples · AID test images · one frozen baseline'
    else:
        class_rows=['Forest','Desert','Park']
        title='Intra-class feature-distance quantiles · AID test images'
    tile_w=250;tile_h=215;header_h=80; side_w=190
    sheet=Image.new('RGB',(side_w+4*tile_w,header_h+len(class_rows)*tile_h),'white')
    dr=ImageDraw.Draw(sheet);dr.text((20,18),title,fill='black')
    dr.text((20,44),run_id,fill='black')
    for j,label in enumerate(selects):dr.text((side_w+j*tile_w+10,62),label,fill='black')
    report=[]
    for i,cl in enumerate(class_rows):
        dr.text((10,header_h+i*tile_h+28),cl,fill='black')
        for j,sel in enumerate(selects):
            match=sub[(sub.class_name.str.casefold()==cl.casefold())&(sub.selection==sel)]
            if len(match)!=1:raise ValueError(f'Pemilihan {cl}/{sel} harus tepat satu, found {len(match)}')
            r=match.iloc[0]
            relative=Path(str(r.sample_id).replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Sample ID tidak aman')
            path=(root/relative).resolve()
            if not path.is_relative_to(root):raise ValueError(f'Path keluar AID root: {relative}')
            if not path.is_file():raise FileNotFoundError(f'Citra AID asli tidak ditemukan: {path}')
            with Image.open(path) as src:
                thumb=ImageOps.fit(src.convert('RGB'),(210,180))
            x=side_w+j*tile_w+6;y=header_h+i*tile_h+10
            sheet.paste(thumb,(x,y))
            dr.text((x,y+183),f"d={float(r.cosine_distance_to_centroid):.3f}",fill='black')
            report.append({'run_id':run_id,'class_name':cl,'selection':sel,'sample_id':str(r.sample_id),
                           'cosine_distance_to_centroid':float(r.cosine_distance_to_centroid)})
    sheet.save(output_path,dpi=(300,300))
    return report


def _figure_evidence_warning(status, filename, detail):
    return {'name':filename,'status':status,'detail':detail}


def build_figures_bab4(e, chart_meta, figure_dir, project_root=None, aid_root=None, contact_run='aid_efc_t5_r18_none_seed1993', raw_sources=None):
    figure_dir=Path(figure_dir);figure_dir.mkdir(parents=True,exist_ok=True)
    project_root=Path(project_root) if project_root else None
    raw_sources=raw_sources or []
    fig_recs=[]
    paired=e['paired_attention_effects.csv'].copy()
    idx={(r.method,r.backbone,r.protocol,r.attention):r for r in paired.itertuples(index=False)}
    rows=[(m,b,p) for m,b,p in product(METHODS_PUB,BACKBONES_PUB,PROTOCOLS_PUB)]
    rowlabels=[f'{METHOD_NAMES_PUB[m]:<7} / {b.upper()} / {p}' for m,b,p in rows]
    for no,col,title,reverse in [
        (1,'delta_aia_pp','Gambar 4.1 — ΔAIA vs None across 60 fair pairs',False),
        (2,'delta_lta_pp','Gambar 4.2 — ΔLTA vs None across 60 fair pairs',False),
        (3,'delta_fr_pp','Gambar 4.3 — ΔFR vs None (negative = less forgetting)',True),
    ]:
        mat=np.array([[float(getattr(idx[m,b,p,a],col)) for a in ATTNS_PUB] for m,b,p in rows])
        dst=figure_dir/f'Gambar_4_{no:02d}.png'
        _plot_heatmap(mat,rowlabels,['ECA','CBAM','ProCBAM'],title,dst,reverse=reverse)
        fig_recs.append(_figure_evidence_warning('generated',dst.name,f'{mat.size} valid pair values; 300 dpi'))

    # 4.4: 20-condition macro average per attention (not task-weighted).
    t9=chart_meta['table_4_9']
    y=np.arange(3);width=0.36
    fig,ax=plt.subplots(figsize=(8.5,5),constrained_layout=True)
    old=t9['Mean ΔOld Accuracy (pp)'].to_numpy(float);new=t9['Mean ΔNew Accuracy (pp)'].to_numpy(float)
    ax.bar(y-width/2,old,width,label='Old-class accuracy')
    ax.bar(y+width/2,new,width,label='New-class accuracy')
    ax.set_xticks(y,t9.Attention);ax.axhline(0,color='black',lw=0.8)
    ax.set_ylabel('Mean change vs None (percentage points)')
    ax.set_title('Gambar 4.4 — Mean ΔOld vs Mean ΔNew Accuracy')
    ax.legend();ax.grid(axis='y',alpha=.2)
    dst=figure_dir/'Gambar_4_04.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    fig_recs.append(_figure_evidence_warning('generated',dst.name,'Macro-average across 20 fair pairs per attention; 300 dpi'))

    # 4.5: exactly same 8 representative extremes as Tabel 4.11.
    extreme=chart_meta['table_4_11'].copy()
    labels=[f"{r.Class} · {METHOD_NAMES_PUB[r.method]} {r.backbone.upper()} {r.protocol} {r.attention.upper()}" for r in extreme.itertuples(index=False)]
    vals=extreme['ΔAccuracy (pp)'].to_numpy(float)
    pos=np.argsort(vals)
    fig,ax=plt.subplots(figsize=(12,5.2),constrained_layout=True)
    ys=np.arange(len(vals))
    ax.barh(ys,vals[pos],color=['tab:green' if v>0 else 'tab:red' for v in vals[pos]])
    ax.set_yticks(ys,[labels[i] for i in pos],fontsize=8)
    ax.axvline(0,color='black',lw=1);ax.set_xlabel('Δ final class accuracy (pp)')
    ax.set_title('Gambar 4.5 — Four highest increases and four largest decreases')
    ax.grid(axis='x',alpha=.2)
    dst=figure_dir/'Gambar_4_05.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    fig_recs.append(_figure_evidence_warning('generated',dst.name,'Same eight cases as Tabel 4.11; 300 dpi'))

    # 4.6: representative feature space baseline with 30 real labels.
    baseline_run='aid_efc_t10_r18_none_seed1993'
    labels30=_label_map(e)
    sims=e['feature_space/inter_class_similarity.csv']
    s=sims[sims.run_id==baseline_run]
    if len(s)!=435:raise ValueError(f'Expected 435 similarities for {baseline_run}, got {len(s)}')
    cm=np.full((30,30),np.nan)
    for r in s.itertuples(index=False):
        i,j=int(r.class_i_id),int(r.class_j_id)
        cm[i,j]=cm[j,i]=float(r.centroid_cosine_similarity)
    if not np.isfinite(cm[~np.eye(30,dtype=bool)]).all():raise ValueError('Matrix similarity missing class-pair entries')
    fig,ax=plt.subplots(figsize=(12,11),constrained_layout=True)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(cm,cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
    ax.set_xticks(range(30),labels30,rotation=90,fontsize=6.5)
    ax.set_yticks(range(30),labels30,fontsize=6.5)
    ax.set_title(f'Gambar 4.6 — Centroid cosine similarity · {baseline_run}',pad=12)
    fig.colorbar(im,ax=ax,label='Cosine similarity (dimensionless)',fraction=.04)
    dst=figure_dir/'Gambar_4_06.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    fig_recs.append(_figure_evidence_warning('generated',dst.name,'435 unordered class pairs; diagonal masked; 300 dpi'))

    # 4.7: positive descriptive association, reference rho is precomputed.
    fc=e['feature_space/feature_confusion_join.csv'];df=fc[fc.run_id==baseline_run].copy()
    corr=e['feature_space/feature_correlation_summary.csv']
    rho=corr[(corr.run_id==baseline_run)&(corr.analysis_type=='similarity_vs_symmetric_confusion')]
    rval=float(rho.iloc[0].spearman_rho) if len(rho)==1 else np.nan
    fig,ax=plt.subplots(figsize=(8.5,6),constrained_layout=True)
    ax.scatter(df.centroid_cosine_similarity,df.symmetric_confusion_rate_pct,s=17,alpha=.4,label='All class pairs')
    colorlist=['tab:red','tab:orange','tab:purple']
    for (cl1,cl2),color in zip([('BareLand','Desert'),('Stadium','Playground'),('Resort','Park')],colorlist):
        subset=df[((df.class_i_name==cl1)&(df.class_j_name==cl2))|((df.class_i_name==cl2)&(df.class_j_name==cl1))]
        if len(subset)!=1:raise ValueError(f'Literature pair {cl1}↔{cl2} not found for scatter')
        r=subset.iloc[0]
        ax.scatter([r.centroid_cosine_similarity],[r.symmetric_confusion_rate_pct],s=82,marker='D',c=color,label=f'{cl1} ↔ {cl2}',edgecolors='black',linewidths=.3)
    ax.set_xlabel('Centroid cosine similarity (dimensionless)')
    ax.set_ylabel('Symmetric mutual confusion rate (%)')
    ax.set_title(f'Gambar 4.7 — Similarity vs Confusion · {baseline_run}\nSpearman ρ = {rval:+.3f} · descriptive, n=435 class pairs')
    ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    dst=figure_dir/'Gambar_4_07.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    fig_recs.append(_figure_evidence_warning('generated',dst.name,'All 435 class pairs; literature pair markers; 300 dpi'))

    # 4.8/4.9 copy verified existing contact sheets or create from real AID images.
    images_root=Path(aid_root).expanduser() if aid_root else None
    if images_root is None and project_root:
        for rel in ['datasets/raw/AID','datasets/AID','data/AID']:
            if (project_root/rel).is_dir():images_root=project_root/rel;break
    for no,kind,basename in [(8,'inter','inter_class_contact_sheet.png'),(9,'intra','intra_class_contact_sheet.png')]:
        dst=figure_dir/f'Gambar_4_{no:02d}.png'
        candidates=[]
        if project_root:
            candidates=[project_root/'outputs/feature_analysis/figures'/basename,
                        project_root/'outputs/analysis/figures'/basename,
                        project_root/'outputs/feature_analysis'/basename,
                        project_root/'outputs/feature_analysis/figures'/dst.name,
                        project_root/'outputs/analysis/figures'/dst.name]
        copied=False
        for src in candidates:
            if src.is_file() and src.resolve()!=dst.resolve():
                shutil.copyfile(src,dst);copied=True;break
        if copied:
            fig_recs.append(_figure_evidence_warning('copied_existing',dst.name,f'Original AID contact sheet from {src}'))
        elif dst.is_file() and (figure_dir/f'Gambar_4_{no:02d}_sample_provenance.csv').is_file():
            fig_recs.append(_figure_evidence_warning('generated_previously',dst.name,
                'Original AID contact sheet created in earlier run with sample provenance CSV'))
        elif images_root and images_root.is_dir():
            try:
                selected=_safe_contact_sheet_from_metadata(e,dst,images_root,contact_run,kind)
                pd.DataFrame(selected).to_csv(figure_dir/f'Gambar_4_{no:02d}_sample_provenance.csv',index=False,encoding='utf-8-sig')
                fig_recs.append(_figure_evidence_warning('generated',dst.name,f'Original AID images, {len(selected)} tiles, selection {contact_run}'))
            except Exception as ex:
                fig_recs.append(_figure_evidence_warning('missing_source',dst.name,f'Cannot create contact sheet: {type(ex).__name__}: {ex}'))
        else:
            fig_recs.append(_figure_evidence_warning('missing_source',dst.name,'Perlu AID_ROOT atau file contact sheet PNG asli; tidak membuat gambar sintetis.'))

    # 4.10: three independent axes (similarity / dispersion unitless, confusion in pp).
    t14=chart_meta['table_4_14']
    pairs_defs=[('Mean ΔCentroid Similarity','Δ centroid cosine similarity'),
                ('Mean ΔIntra Dispersion','Δ intra-class cosine dispersion'),
                ('Mean ΔMutual Confusion (pp)','Δ symmetric confusion rate (pp)')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for ax,(key,name) in zip(axes,pairs_defs):
        data=t14[key].to_numpy(float)
        ax.bar(t14.Attention,data,color=['tab:green' if v<0 else 'tab:red' for v in data])
        ax.axhline(0,color='black',linewidth=1)
        ax.set_title(name,fontsize=10)
        ax.grid(axis='y',alpha=.2)
        ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Gambar 4.10 — Mean feature-space effect across 20 fair pairs / attention\nNegative change = lower similarity, dispersion, or mutual confusion',fontsize=11)
    dst=figure_dir/'Gambar_4_10.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
    fig_recs.append(_figure_evidence_warning('generated',dst.name,'Three scales kept separate; 300 dpi'))

    # 4.11-12 reconstructed from **canonical count columns**, verified vs LTA and support.
    for no,method,protocol,backbone,att in [(11,'fecam','T10','r18','cbam'),(12,'pass','T10','r18','cbam')]:
        rid_none=f'aid_{method}_{protocol.lower()}_{backbone}_none_seed1993'
        rid_att=f'aid_{method}_{protocol.lower()}_{backbone}_{att}_seed1993'
        base=_build_cm_from_csv(e,rid_none,raw_sources)
        after=_build_cm_from_csv(e,rid_att,raw_sources)
        if not np.array_equal(base.sum(axis=1),after.sum(axis=1)):
            raise ValueError(f'Confusion pair test supports differ: {rid_none} vs {rid_att}')
        norm_base=base/base.sum(axis=1,keepdims=True)*100
        norm_after=after/after.sum(axis=1,keepdims=True)*100
        fig,axes=plt.subplots(1,2,figsize=(15.5,7),constrained_layout=True)
        for ax,arr,ttl in zip(axes,[norm_base,norm_after],['None (baseline)','CBAM (attention)']):
            im=ax.imshow(arr,cmap='Blues',vmin=0,vmax=100)
            ax.set_xticks(range(30),labels30,rotation=90,fontsize=5)
            ax.set_yticks(range(30),labels30,fontsize=5)
            ax.set_title(ttl);ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
            fig.colorbar(im,ax=ax,fraction=.045,pad=.02,label='Within true-class row (%)')
        fig.suptitle(f'Gambar 4.{no} — {METHOD_NAMES_PUB[method]} {backbone.upper()} {protocol}: confusion matrix (None vs CBAM)',fontsize=12)
        dst=figure_dir/f'Gambar_4_{no:02d}.png';fig.savefig(dst,dpi=300,bbox_inches='tight');plt.close(fig)
        fig_recs.append(_figure_evidence_warning('generated',dst.name,'Exactly reconstructed from TP + off-diagonal error counts, validated against final supports and LTA; 300 dpi'))
    return fig_recs



def build_optional_figures_bab4(e, figure_dir):
    """Extra visuals (not counted as mandatory Gambar 4.1–4.12)."""
    figure_dir=Path(figure_dir)
    recs=[]
    task=e['task_accuracy_long.csv']
    for method,prot,backbone,att in [('fecam','T10','r18','cbam'),('pass','T10','r18','cbam')]:
        s=task[(task.method==method)&(task.protocol==prot)&(task.backbone==backbone)&task.attention.isin(['none',att])]
        if len(s)!=20:raise ValueError('Kurva tambahan memerlukan 10 task × 2 metode attention/base')
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for a,lab in [('none','None'),(att,att.upper())]:
            g=s[s.attention==a].sort_values('task_id')
            ax.plot(g.task_id+1,g.top1_accuracy_pct,marker='o',linewidth=1.7,label=lab)
        ax.set_xticks(range(1,11));ax.set_ylim(0,100)
        ax.set_xlabel('Incremental task (1-based)');ax.set_ylabel('Joint Top-1 accuracy (%)')
        ax.grid(alpha=.25);ax.legend()
        ax.set_title(f'Optional — Accuracy curve {METHOD_NAMES_PUB[method]} {backbone.upper()} {prot}: None vs CBAM')
        out=figure_dir/f'Opsional_accuracy_curve_{method}_{prot}_{backbone}_{att}.png'
        fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)
        recs.append({'name':out.name,'status':'generated','detail':'Per-task production Top-1, 300 dpi'})

    f=e['class_forgetting_attention_delta.csv'];f=f[f.support>=10]
    good=f[f.delta_forgetting_pp<0].nsmallest(10,'delta_forgetting_pp')
    bad=f[f.delta_forgetting_pp>0].nlargest(10,'delta_forgetting_pp')
    tmp=pd.concat([good,bad]).sort_values('delta_forgetting_pp')
    fig,ax=plt.subplots(figsize=(12,8),constrained_layout=True)
    lab=[f'{r.class_name}: {r.method} {r.backbone.upper()} {r.protocol} {r.attention}' for r in tmp.itertuples(index=False)]
    vals=tmp.delta_forgetting_pp.to_numpy(float)
    ax.barh(range(len(tmp)),vals,color=['tab:green' if v<0 else 'tab:red' for v in vals])
    ax.set_yticks(range(len(tmp)),lab,fontsize=7.5)
    ax.axvline(0,color='black',linewidth=1);ax.set_xlabel('Δ per-class forgetting (pp)')
    ax.set_title('Optional — Largest decreases/increases in per-class forgetting')
    out=figure_dir/'Opsional_class_forgetting_diverging.png'
    fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)
    recs.append({'name':out.name,'status':'generated','detail':'10 best and 10 worst across 60 fair pairs, 300 dpi'})

    baseline_run='aid_efc_t10_r18_none_seed1993'
    d=e['feature_space/intra_class_dispersion.csv']
    d=d[d.run_id==baseline_run]
    if len(d)!=30:raise ValueError('Optional intra dispersion charts require 30 AID classes')
    corr=e['feature_space/feature_correlation_summary.csv']
    for name,col,unit,analysis_type in [
      ('final_error','final_class_error_rate_pct','Final class error rate (%)','dispersion_vs_final_error'),
      ('forgetting','class_forgetting_pp','Per-class forgetting (pp)','dispersion_vs_class_forgetting')]:
        res=corr[(corr.run_id==baseline_run)&(corr.analysis_type==analysis_type)]
        rho=float(res.iloc[0].spearman_rho) if len(res)==1 else float('nan')
        fig,ax=plt.subplots(figsize=(7.5,5),constrained_layout=True)
        ax.scatter(d.mean_cosine_distance_to_centroid,d[col],alpha=.72,s=35)
        ax.set_xlabel('Mean intra-class cosine distance (dimensionless)')
        ax.set_ylabel(unit);ax.grid(alpha=.2)
        ax.set_title(f'Optional — Intra dispersion vs {name}\n{baseline_run} · Spearman ρ={rho:+.3f} (descriptive)')
        out=figure_dir/f'Opsional_intra_dispersion_vs_{name}.png'
        fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)
        recs.append({'name':out.name,'status':'generated','detail':'30 classes, 300 dpi'})
    return recs

def run_bab4_publication(bundle_path, output_root, *, project_root=None, aid_root=None, raw_sources=None, generate_optional=True):
    """Generate Tabel 4.1–4.16 + Gambar 4.1–4.12, record missing contact sheets."""
    e=load_bab4_evidence(bundle_path)
    output_root=Path(output_root).resolve()
    table_dir=output_root/'tables';figure_dir=output_root/'figures'
    output_root.mkdir(parents=True,exist_ok=True)
    tables, chart_meta=build_tables_bab4(e,table_dir)
    figures=build_figures_bab4(e,chart_meta,figure_dir,project_root=project_root,aid_root=aid_root,raw_sources=raw_sources)
    optional=build_optional_figures_bab4(e,figure_dir) if generate_optional else []
    manifest={'generated_at':datetime.now(timezone.utc).isoformat(), 'source_zip':str(Path(bundle_path).resolve()),
              'scope':'AID only, seed 1993', 'models':80,'fair_pairs':60,
              'table_count':len(tables),'figure_generated_or_copied':sum(f['status'] in ('generated','copied_existing','generated_previously') for f in figures),
              'figure_missing':sum(f['status'] not in ('generated','copied_existing','generated_previously') for f in figures),
              'tables':tables,'figures':figures,'optional_figures':optional,
              'caveats':['AIA/LTA/FR/BWT sourced verbatim from audited CSV, no metric redefinition.',
                         'Feature-space similarity is not visual similarity; negative delta cosine distance is a different unit than pp.',
                         'Only a single seed (1993). Spearman correlations across dependent class pairs are descriptive.',
                         'sample_identity_match=label_order_only: original canonical sample IDs were not archived in prediction artifacts.',
                         'Gambar 4.11–4.12 reconstructed from exact per-class TP / off-diagonal error counts; validates LTA and 2000 support.',
                         'Contact sheets require real original AID images or an existing validated PNG; never substitute synthetic images.']}
    (output_root/'bab4_publication_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(figures).to_csv(output_root/'bab4_figures_status.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tables).to_csv(output_root/'bab4_tables_status.csv',index=False,encoding='utf-8-sig')
    return manifest
