import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, NormalizeIntensityd, 
    CropForegroundd, Resized, ToTensord, MapTransform, MapLabelValueD
)

class AmosNormalizeD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        filename = os.path.basename(d["image_meta"])
        try:
            case_id = int(filename.split('_')[1].split('.')[0])
        except:
            case_id = 1
            
        if "modality" in d:
             # Explicit modality from frontend/API
             # if modality is ct -> use CT scaler
             # else (mri) -> use MRI scaler
             if d["modality"] == "ct":
                 scaler = ScaleIntensityRanged(
                    keys=self.keys, a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                )
             else:
                 # Assumes 'mri' or default behavior for non-ct explicit modality
                 scaler = NormalizeIntensityd(keys=self.keys, nonzero=True)
        else:
            # Fallback to existing filename-based logic
            if 1 <= case_id <= 500:
                scaler = ScaleIntensityRanged(
                    keys=self.keys, a_min=-175, a_max=250,
                    b_min=0.0, b_max=1.0, clip=True,
                )
            else:
                scaler = NormalizeIntensityd(keys=self.keys, nonzero=True)
        
        return scaler(d)

def get_amos_datalist(data_dir, mode="Tr"):
    image_path = os.path.join(data_dir, f"images{mode}", "*.nii.gz")
    label_path = os.path.join(data_dir, f"labels{mode}", "*.nii.gz")
    image_files = sorted(glob.glob(image_path))
    label_files = sorted(glob.glob(label_path))
    
    data_dicts = []
    for img, lbl in zip(image_files, label_files):
        data_dicts.append({"image": img, "label": lbl, "image_meta": img})
    return data_dicts

def get_transforms():
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        AmosNormalizeD(keys=["image"]),
        
        # REMAPPING LOGIC FOR 8 CLASSES:
        # We take original labels 1-15 and map them to our new 0-7 sequence.
        # Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # Target:   [1, 2, 3, 4, 0, 6, 7, 0, 0,  5,  0,  0,  0,  0,  0]
        # Note: 5 (Esophagus) becomes 0, 10 (Pancreas) becomes 5.
        MapLabelValueD(
            keys=["label"],
            orig_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            target_labels=[1, 2, 3, 4, 0, 6, 7, 0, 0, 5, 0, 0, 0, 0, 0]
        ),
        
        CropForegroundd(keys=keys, source_key="image"),
        Resized(keys=keys, spatial_size=[128, 128, 64]),
        ToTensord(keys=keys),
    ])