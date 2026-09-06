import json
from config import DATA_ROOT, SPLIT_PATH, WINDOW, SEED
from data import patient_dirs, load_splits, CTVGTVCase, sample_epoch_plans
splits = load_splits(SPLIT_PATH)
paths = {p.name:p for p in patient_dirs(DATA_ROOT, 'train')}
case = CTVGTVCase(paths[splits['folds'][0]['train'][0]])
plans = sample_epoch_plans([case], 1, SEED, WINDOW)
assert len(plans) == 5
for plan in plans:
    video, prompts = case.video(plan['source_indices'])
    assert len(video.frames) == 8
    assert tuple(prompts.shape) == (8, 1, 1024, 1024)
print(json.dumps({'patient':case.patient,'window_categories':[p['category'] for p in plans],'target_frames':8,'prompt_shape':[8,1,1024,1024],'folds':len(splits['folds'])}, indent=2))