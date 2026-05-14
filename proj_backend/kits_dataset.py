import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, CropForegroundd, 
    RandCropByPosNegLabeld, ToTensord, EnsureTyped, SpatialPadd
)
from monai.data import CacheDataset, DataLoader

def get_kits_dataloader(data_dir="./data", batch_size=1, phase="train"):
    sub_folder = "trainData" if phase in ["train", "val"] else "testData"
    root_path = os.path.join(data_dir, sub_folder)
    case_dirs = sorted(glob.glob(os.path.join(root_path, "case_*")))
    
    data_dicts = []
    for d in case_dirs:
        img = glob.glob(os.path.join(d, "imaging.nii.gz"))
        lbl = glob.glob(os.path.join(d, "segmentation.nii"))
        if img:
            item = {"image": img[0]}
            if phase != "test" and lbl: item["label"] = lbl[0]
            data_dicts.append(item)

    if phase in ["train", "val"]:
        split = int(len(data_dicts) * 0.8)
        data_dicts = data_dicts[:split] if phase == "train" else data_dicts[split:]

    transforms = Compose([
        LoadImaged(keys=["image", "label"] if phase != "test" else ["image"]),
        EnsureChannelFirstd(keys=["image", "label"] if phase != "test" else ["image"]),
        Orientationd(keys=["image", "label"] if phase != "test" else ["image"], axcodes="RAS"),
        Spacingd(keys=["image", "label"] if phase != "test" else ["image"], 
                 pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest" if phase != "test" else "bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"] if phase != "test" else ["image"], source_key="image"),
        EnsureTyped(keys=["image", "label"] if phase != "test" else ["image"]),
        SpatialPadd(keys=["image", "label"] if phase != "test" else ["image"], spatial_size=(128, 128, 128)),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(128, 128, 128), pos=1, neg=1, num_samples=2 
        ) if phase == "train" else EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])

    # 64GB RAM Optimization
    ds = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=8)
    return DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=4)

if __name__ == "__main__":
    print("🔍 Testing Dataset pairing...")
    try:
        loader = get_kits_dataloader(batch_size=1, phase="train")
        print(f"✅ Found {len(loader.dataset)} training cases!")
        batch = next(iter(loader))
        print(f"✅ Image shape: {batch['image'].shape}")
        print(f"✅ Label values: {torch.unique(batch['label'])}")
    except Exception as e:
        print(f"❌ Dataset Error: {e}")