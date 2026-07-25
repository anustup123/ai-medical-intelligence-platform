# Dataset Instructions

This project uses the **Chest X-Ray Images (Pneumonia)** dataset (Kermany et al.,
distributed on Kaggle by Paul Mooney).

- Kaggle page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Size: ~1.2 GB, 5,863 JPEG images in 2 classes (NORMAL, PNEUMONIA)

## Steps to download

1. Create a free Kaggle account if you don't have one.
2. Go to the dataset page above and click **Download** (or use the Kaggle CLI):
   ```bash
   pip install kaggle
   # Place your kaggle.json API token in ~/.kaggle/kaggle.json first
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```
3. After unzipping, make sure the structure looks EXACTLY like this
   (the Kaggle zip sometimes nests an extra folder — flatten it if so):
   ```
   data/chest_xray/
       train/
           NORMAL/       (~1341 images)
           PNEUMONIA/    (~3875 images)
       val/
           NORMAL/       (8 images)
           PNEUMONIA/    (8 images)
       test/
           NORMAL/       (234 images)
           PNEUMONIA/    (390 images)
   ```
4. Note: the official `val/` split is tiny (16 images total), which makes
   validation metrics noisy. A common, recommended fix (optional but
   encouraged for your report) is to re-split: merge `train/` + `val/`,
   then re-split 85/15 with stratification so validation is more reliable.
   A helper script for this is below.

## Optional: better train/val split

If you want a more statistically meaningful validation set, run this once:

```python
import os, shutil, random
from pathlib import Path

random.seed(42)
base = Path("data/chest_xray")
for cls in ["NORMAL", "PNEUMONIA"]:
    train_files = list((base / "train" / cls).glob("*.jpeg"))
    val_files = list((base / "val" / cls).glob("*.jpeg"))
    all_files = train_files + val_files
    random.shuffle(all_files)

    split_idx = int(len(all_files) * 0.85)
    new_train, new_val = all_files[:split_idx], all_files[split_idx:]

    # Clear and repopulate
    for f in train_files + val_files:
        os.remove(f)
    for f in new_train:
        shutil.copy(f, base / "train" / cls / f.name)
    for f in new_val:
        shutil.copy(f, base / "val" / cls / f.name)

print("Re-split complete.")
```

This dataset is NOT included in this repository (it's large and Kaggle's
license requires downloading it directly from the source) — hence why
`data/chest_xray/` is listed in `.gitignore`.
