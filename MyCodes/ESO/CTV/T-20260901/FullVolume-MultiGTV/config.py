from pathlib import Path
ROOT=Path('/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/FullVolume-MultiGTV')
RESULT_ROOT=Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/FullVolume-MultiGTV')
DATA_ROOT=Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/datanii')
SPLIT_PATH=ROOT.parent/'shared_splits.json'
SAM2_ROOT=Path('/home/wusi/SAM2');MODEL_CFG='configs/sam2.1/sam2.1_hiera_l.yaml';CHECKPOINT=SAM2_ROOT/'checkpoints/sam2.1_hiera_large.pt'
TARGET_SPACING_XYZ=(1.25,1.25,5.0);IMAGE_SIZE=1024;SEED=20260905;MAX_EPOCHS=100;PATIENCE=10;LR=1e-4;WEIGHT_DECAY=1e-4;WINDOW=8;WINDOWS_PER_CASE=5
LORA_RANK=4;LORA_ALPHA=16.;LORA_DROPOUT=.1;LAMBDA_GRID=(0.,.01,.05,.10,.20)