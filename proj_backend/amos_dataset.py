import os
import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, NormalizeIntensityd, 
    CropForegroundd, Resized, ToTensord, MapTransform, MapLabelValueD,
    RandFlipd, RandRotate90d
)
from monai.data import CacheDataset, DataLoader

class AmosNormalizeD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        # Handle filename-based ID extraction
        filename = os.path.basename(d.get("image_meta", "amos_0001.nii.gz"))
        try:
            case_id = int(filename.split('_')[1].split('.')[0])
        except:
            case_id = 1
            
        # Modality logic: Priority to explicit 'modality' key (for Frontend)
        # Fallback to Case ID range
        modality = d.get("modality", "ct" if 1 <= case_id <= 500 else "mri")

        if str(modality).lower() == "ct":
            scaler = ScaleIntensityRanged(
                keys=self.keys, a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True,
            )
        else:
            scaler = NormalizeIntensityd(keys=self.keys, nonzero=True)
        
        return scaler(d)

def get_dataloader(data_dir="./data", batch_size=1, phase="train"):
    mode = "Tr" if phase == "train" else "Va"
    images = sorted(glob.glob(os.path.join(data_dir, f"images{mode}", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, f"labels{mode}", "*.nii.gz")))
    
    data_dicts = [{"image": i, "label": l, "image_meta": i} for i, l in zip(images, labels)]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        
        # 1. Your Modality Logic
        AmosNormalizeD(keys=["image"]),
        
        # 2. Your REMAPPING LOGIC FOR 8 CLASSES
        MapLabelValueD(
            keys=["label"],
            orig_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            target_labels=[1, 2, 3, 4, 0, 6, 7, 0, 0, 5, 0, 0, 0, 0, 0]
        ),
        
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=[96, 96, 96]),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0) if phase=="train" else Compose([]),
        ToTensord(keys=["image", "label"]),
    ])

    ds = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4)
    return DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=0)

if __name__ == "__main__":
    try:
        print("Testing Training Dataloader...")
        train_loader = get_dataloader(data_dir="./data", batch_size=1, phase="train")
        print(f"Total training samples: {len(train_loader.dataset)}")
        
        check_data = next(iter(train_loader))
        print(f"Image shape: {check_data['image'].shape}")
        print(f"Label shape: {check_data['label'].shape}")
        print(f"Unique label values after remapping: {torch.unique(check_data['label'])}")
        
        print("\nTesting Validation Dataloader...")
        val_loader = get_dataloader(data_dir="./data", batch_size=1, phase="val")
        print(f"Total validation samples: {len(val_loader.dataset)}")
        print("\n--- ALL CHECKS PASSED. You are ready to train! ---")
        
    except Exception as e:
        print(f"\n--- ERROR FOUND ---")
        print(e)
