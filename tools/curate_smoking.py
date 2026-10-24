"""Create a source-preserving four-class dataset; 'smoking' boxes label cigarettes, not actors."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml

MAPPING={0:0,1:1,2:2,3:3,4:0}
NAMES=['cigarette','person','smoke','vape']


def curate(audit_dir,out):
    audit_dir=Path(audit_dir).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    counts={};links=Counter()
    config={'path':str(out),'nc':4,'names':NAMES}
    for source_split,target_key in [('train','train'),('valid','val'),('test','test')]:
        images=(out/source_split/'images');labels=(out/source_split/'labels')
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        listed=Path(audit_dir/(source_split+'.txt')).read_text(encoding='utf-8').splitlines()
        output_paths=[];boxes=Counter()
        for value in listed:
            if not value.strip():continue
            source=Path(value)
            target=images/source.name
            if not target.exists():
                try:os.link(source,target);links['hardlinks']+=1
                except OSError:shutil.copy2(source,target);links['copies']+=1
            annotation=source.parent.parent/'labels'/(source.stem+'.txt')
            result=[];seen=set()
            for line in annotation.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                old,*coords=line.split()
                cls=MAPPING[int(old)]
                key=(cls,*map(float,coords))
                if key in seen:continue
                seen.add(key)
                result.append(str(cls)+' '+' '.join(coords))
                boxes[NAMES[cls]]+=1
            (labels/(source.stem+'.txt')).write_text('\n'.join(result)+('\n' if result else ''),encoding='utf-8')
            output_paths.append(str(target))
        listing=out/(source_split+'.txt');listing.write_text('\n'.join(output_paths)+'\n',encoding='utf-8')
        config[target_key]=str(listing)
        counts[source_split]={'images':len(output_paths),'boxes':dict(boxes)}
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    manifest={'source_url':'https://universe.roboflow.com/richie-lab/smoking-tasfx/dataset/4','author':'Richie Lab','license':'CC BY 4.0',
              'source_audit_sha256':hashlib.sha256((audit_dir/'audit.json').read_bytes()).hexdigest(),
              'mapping':MAPPING,'names':NAMES,'counts':counts,'storage':dict(links),
              'rationale':'Training annotation inspection shows the smoking class boxes surround cigarettes, not people. Merge semantic aliases before fitting; test labels use the same fixed mapping. Original files are unchanged.'}
    (out/'curation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return out/'data.yaml'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(curate(args.audit,args.out))
