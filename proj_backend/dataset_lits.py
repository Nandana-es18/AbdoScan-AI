import os
import glob
import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, CropForegroundd, 
    RandCropByPosNegLabeld, ToTensord, EnsureTyped, SpatialPadd
)
from monai.data import CacheDataset, DataLoader

def get_lits_dataloader(data_dir="./data", batch_size=1, phase="train"):
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "liver_*.nii")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "liver_*.nii")))
    
    # Verify file pairing and integrity
    valid_dicts = []
    print(f"🔍 Checking integrity of {len(images)} files...")
    for img, lbl in zip(images, labels):
        try:
            # Quick check if file is readable
            nib.load(img)
            valid_dicts.append({"image": img, "label": lbl})
        except Exception as e:
            print(f"⚠️ Skipping corrupted file: {img} | Error: {e}")

    # 80/20 Split for Research
    split = int(len(valid_dicts) * 0.8)
    data_dicts = valid_dicts[:split] if phase == "train" else valid_dicts[split:]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(128, 128, 128), pos=1, neg=1, num_samples=2 
        ) if phase == "train" else EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

    # 64GB RAM Optimization: cache_rate 0.8 leaves room for system overhead
    ds = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.8, num_workers=4)
    return DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=2)

if __name__ == "__main__":
    loader = get_lits_dataloader()
    print(f"✅ Dataloader ready with {len(loader.dataset)} cases.")