from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();best=None
 for d in sorted(a.root.glob('fold_*')):
  rows=list(csv.DictReader((d/'epoch_metrics.csv').open()));r=max(rows,key=lambda x:float(x['validation_dice_original_space']))
  x={'fold':d.name,'epoch':int(r['epoch']),'validation_dice':float(r['validation_dice_original_space']),'checkpoint':str(d/'best.pth')}
  if best is None or x['validation_dice']>best['validation_dice']:best=x
 if best is None:raise RuntimeError('no completed folds')
 (a.root/'selected_best_fold.json').write_text(json.dumps(best,indent=2));print(json.dumps(best,indent=2))
if __name__=='__main__':main()