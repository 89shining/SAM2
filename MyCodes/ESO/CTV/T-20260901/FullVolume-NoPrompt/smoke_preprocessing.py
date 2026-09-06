from __future__ import annotations
import json
from config import DATA_ROOT,SPLIT_PATH,TARGET_SPACING_XYZ
from data import load_splits,patient_dirs,CTVCase,sample_epoch_plans

def main():
 s=load_splits(SPLIT_PATH); lookup={p.name:p for p in patient_dirs(DATA_ROOT,'train')}; f=s['folds'][0]
 assert set(f['train']).isdisjoint(f['val']) and len(f['train'])==72 and len(f['val'])==18
 c=CTVCase(lookup[f['train'][0]]);assert c.rgb.shape[1]==3 and c.ctv.dtype==bool;assert c.reference.GetSpacing()==TARGET_SPACING_XYZ
 plans=sample_epoch_plans([c],1,20260905);assert len(plans)==5 and {x['category'] for x in plans}=={'internal','superior_crossing','inferior_crossing','superior_negative','inferior_negative'}
 restored_probability=c.restore_probability_to_original(c.ctv.astype('float32'));assert restored_probability.shape==__import__('SimpleITK').GetArrayFromImage(c.original_ctv).shape; restored=restored_probability>0.5
 print(json.dumps({'patient':c.patient,'original_size':c.original_ct.GetSize(),'original_spacing':c.original_ct.GetSpacing(),'resampled_size':c.reference.GetSize(),'resampled_spacing':c.reference.GetSpacing(),'plans':plans},indent=2))
if __name__=='__main__':main()