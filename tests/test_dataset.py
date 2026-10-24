from pathlib import Path
import tempfile
import unittest
import json
import cv2
import numpy as np
import yaml
from tools.audit_dataset import audit
from tools.curate_smoking import curate


class DatasetTests(unittest.TestCase):
    def test_holdout_duplicates_excluded_and_alias_mapping_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir()
            (root/'data.yaml').write_text(yaml.safe_dump({'names':['Cigarette','Person','Smoke','Vape','smoking']}),encoding='utf-8')
            a=np.random.default_rng(1).integers(0,255,(96,96,3),dtype=np.uint8)
            b=np.random.default_rng(2).integers(0,255,(96,96,3),dtype=np.uint8)
            def save(split,name,image,labels):
                images=root/split/'images';annotations=root/split/'labels'
                images.mkdir(parents=True,exist_ok=True);annotations.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(images/(name+'.png')),image)
                path=annotations/(name+'.txt');path.write_text(labels,encoding='utf-8');return path
            save('test','heldout',a,'0 .5 .5 .1 .1\n')
            save('valid','copy',a,'0 .5 .5 .1 .1\n')
            save('train','flipped',a[:,::-1],'4 .5 .5 .1 .1\n')
            original='0 .5 .5 .1 .1\n4 .5 .5 .1 .1\n3 .2 .2 .1 .1\n'
            label=save('train','independent',b,original)
            report=audit(root,Path(directory)/'audit')
            self.assertEqual(report['kept'],{'test':1,'valid':0,'train':1})
            output=curate(Path(directory)/'audit',Path(directory)/'curated')
            config=yaml.safe_load(output.read_text(encoding='utf-8'))
            self.assertEqual(config['names'],['cigarette','person','smoke','vape'])
            self.assertEqual(label.read_text(encoding='utf-8'),original)
            curated=output.parent/'train/labels/independent.txt'
            self.assertEqual(curated.read_text().splitlines(),['0 .5 .5 .1 .1','3 .2 .2 .1 .1'])
