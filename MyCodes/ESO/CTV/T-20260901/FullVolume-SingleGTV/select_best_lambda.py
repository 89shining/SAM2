from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();candidates=[]
 for d in sorted(a.root.glob('lambda_*')):
  rows=list(csv.DictReader((d/'epoch_metrics.csv').open()))
  if not rows:raise RuntimeError(f'no epoch metrics: {d}')
  best=max(rows,key=lambda r:(float(r['validation_dice_original_space']),-float(r['validation_hd95_mm_original_space'])))
  candidates.append({'lambda_rank':float(d.name.split('lambda_',1)[1]),'best_epoch':int(best['epoch']),'validation_dice':float(best['validation_dice_original_space']),'validation_hd95_mm':float(best['validation_hd95_mm_original_space']),'checkpoint':str(d/'best.pth')})
 if len(candidates)!=5:raise RuntimeError(f'expected 5 lambda runs, got {len(candidates)}')
 selected=max(candidates,key=lambda x:(x['validation_dice'],-x['validation_hd95_mm']))
 (a.root/'lambda_selection.json').write_text(json.dumps({'selection_rule':'highest original-space validation Dice3D; tie: lower HD95','candidates':candidates,'selected':selected},indent=2));print(json.dumps(selected,indent=2))
if __name__=='__main__':main()