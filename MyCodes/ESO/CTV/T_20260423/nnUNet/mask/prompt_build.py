import os
from pathlib import Path

A_ROOT = Path('/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset008_EsoCTV73p/nnUNetTrainer__nnUNetPlans__3d_fullres')
B_ROOT = Path('/home/wusi/SAM2/SAM2data/Eso/20260326/datanii')

TRAIN_DST = B_ROOT / 'train_nii'
TEST_DST = B_ROOT / 'test_nii'


def case_id_from_name(name: str) -> str:
    """Extract numeric case id from file names like CTV_006.nii.gz -> 6."""
    stem = name
    if stem.endswith('.nii.gz'):
        stem = stem[:-7]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f'Cannot extract numeric case id from: {name}')
    return str(int(digits))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_train_prompts(src_root: Path):
    files = []
    for i in range(5):
        vdir = src_root / f'fold_{i}' / 'validation'
        if not vdir.is_dir():
            raise FileNotFoundError(f'Missing validation dir: {vdir}')
        files.extend(sorted(vdir.glob('*.nii.gz')))
    return files


def collect_test_prompts(src_root: Path):
    tdir = src_root / 'TestResults_28p_fold1' / 'GTcrop'
    if not tdir.is_dir():
        raise FileNotFoundError(f'Missing test dir: {tdir}')
    return sorted(tdir.glob('*.nii.gz'))


def copy_prompt_files(src_files, dst_root: Path, split_name: str):
    import shutil

    copied = 0
    skipped = 0
    for src in src_files:
        cid = case_id_from_name(src.name)
        dst_case = dst_root / f'p_{cid}'
        if not dst_case.is_dir():
            print(f'[{split_name}] skip (target folder not found): {dst_case}')
            skipped += 1
            continue

        dst_file = dst_case / 'prompt.nii.gz'
        shutil.copy2(src, dst_file)
        print(f'[{split_name}] {src} -> {dst_file}')
        copied += 1

    print(f'[{split_name}] done: copied={copied}, skipped={skipped}')


def main():
    if not A_ROOT.is_dir():
        raise FileNotFoundError(f'A_ROOT not found: {A_ROOT}')
    if not B_ROOT.is_dir():
        raise FileNotFoundError(f'B_ROOT not found: {B_ROOT}')

    ensure_dir(TRAIN_DST)
    ensure_dir(TEST_DST)

    # Train prompt files were already copied previously; keep them unchanged.
    test_files = collect_test_prompts(A_ROOT)

    print(f'Collected test prompt files: {len(test_files)}')

    copy_prompt_files(test_files, TEST_DST, 'test')


if __name__ == '__main__':
    main()
