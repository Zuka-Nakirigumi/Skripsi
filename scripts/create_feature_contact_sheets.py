#!/usr/bin/env python3
"""Create deterministic contact sheets from real sample_path metadata, if accessible.

Requires feature_sample_distances.csv from analyze_feature_space.py; does not
retrieve images over network or guess AID paths. Selection metadata is saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PAIRS = [
    ("Stadium", "Playground"), ("BareLand", "Desert"), ("Resort", "Park"),
    ("Church", "Commercial"), ("Beach", "Port"), ("Square", "Center"),
    ("MediumResidential", "DenseResidential"),
]

def stable_int(value):
    return int(hashlib.sha256(value.encode('utf-8')).hexdigest()[:12],16)

def tile(path, label, size=145):
    panel=Image.new('RGB',(size,size+28),'white')
    try:
        with Image.open(path) as im:
            im=im.convert('RGB')
            im.thumbnail((size,size))
            panel.paste(im,((size-im.width)//2,(size-im.height)//2))
    except Exception:
        ImageDraw.Draw(panel).text((6,58),'Image missing',fill='black')
    ImageDraw.Draw(panel).text((5,size+6),label[:22],fill='black')
    return panel

def resolve_sample(path, project_root):
    if not isinstance(path,str) or not path.strip() or path.lower()=='nan':
        return None
    p=Path(path)
    if not p.is_absolute():
        p=project_root / p
    return p if p.is_file() else None

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--project-root',default='.')
    a.add_argument('--feature-dir',default='outputs/feature_analysis')
    a.add_argument('--run-id',required=True)
    opt=a.parse_args()
    root=Path(opt.project_root).resolve();fd=root/opt.feature_dir
    source=fd/'feature_sample_distances.csv'
    if not source.is_file(): raise FileNotFoundError(source)
    df=pd.read_csv(source)
    df=df[df.run_id==opt.run_id].copy()
    if df.empty: raise ValueError('run_id not found in feature sample distances')
    df['resolved_path']=df.sample_path.map(lambda p: str(resolve_sample(p,root) or ''))
    available=df[df.resolved_path!=''].copy()
    if available.empty:
        print('No accessible sample_path. Contact-sheet generation skipped; no images invented.')
        return
    out=fd/'figures';out.mkdir(exist_ok=True,parents=True)
    selected=[]; thumb_w=145; thumb_h=173
    canvas=Image.new('RGB',(2*4*thumb_w,7*thumb_h),'white')
    for row,(name_a,name_b) in enumerate(PAIRS):
        for col,name in enumerate((name_a,name_b)):
            sub=available[available.class_name.str.lower()==name.lower()].copy()
            if sub.empty:continue
            sub=sub.assign(_order=sub.sample_id.fillna(sub.sample_index.astype(str)).astype(str).map(
                lambda v:stable_int(f'1993|{name}|{v}'))).sort_values('_order')
            for k,(_,s) in enumerate(sub.head(4).iterrows()):
                x=(col*4+k)*thumb_w;y=row*thumb_h
                canvas.paste(tile(s.resolved_path,name),(x,y))
                selected.append({'run_id':opt.run_id,'type':'inter','class':name,'sample_path':s.resolved_path,
                                 'sample_index':int(s.sample_index),'selection':'stable-hash-seed1993'})
    if selected:
        canvas.save(out/'inter_class_contact_sheet.png',dpi=(300,300))
    # Intra-class: select five objectively positioned class distances for 5 highest dispersion classes.
    ranking=available.groupby('class_name').cosine_distance_to_centroid.mean().nlargest(5).index.tolist()
    canvas=Image.new('RGB',(5*thumb_w,5*thumb_h),'white')
    for row,name in enumerate(ranking):
        sub=available[available.class_name==name].sort_values('cosine_distance_to_centroid').reset_index(drop=True)
        for col,(tag,q) in enumerate([('closest',0),('median',.5),('p75',.75),('p90',.9),('farthest',1)]):
            s=sub.iloc[int(round(q*(len(sub)-1)))];path=s.resolved_path
            canvas.paste(tile(path,f'{name} {tag}'),(col*thumb_w,row*thumb_h))
            selected.append({'run_id':opt.run_id,'type':'intra','class':name,'sample_path':path,
                             'sample_index':int(s.sample_index),'selection':tag})
    if ranking:
        canvas.save(out/'intra_class_contact_sheet.png',dpi=(300,300))
    pd.DataFrame(selected).to_csv(fd/'contact_sheet_selected_samples.csv',index=False,encoding='utf-8-sig')
    print('Generated contact sheets:',out)

if __name__=='__main__':main()
