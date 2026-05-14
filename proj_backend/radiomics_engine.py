import numpy as np
from skimage import measure

def extract_radiomics(image, mask, tumor_id=2, spacing=(1.5, 1.5, 1.5)):
    """
    Research-Grade Radiomics with Noise Filtering.
    spacing: (dx, dy, dz) in mm from the NIfTI header.
    """
    tumor_mask = (mask == tumor_id).astype(np.uint8)
    
    # 1. Volume Calculation
    voxel_volume = np.prod(spacing)
    total_voxels = np.sum(tumor_mask)
    volume_mm3 = total_voxels * voxel_volume
    
    # --- CLINICAL NOISE FILTER ---
    # kidney/liver tumors are rarely smaller than 100mm3. 
    # Anything smaller is usually AI segmentation noise.
    if volume_mm3 < 100:
        return None 

    # 2. Intensity Statistics (Heterogeneity)
    tumor_intensities = image[tumor_mask == 1]
    mean_int = np.mean(tumor_intensities)
    std_int = np.std(tumor_intensities)
    heterogeneity = (std_int / mean_int) if mean_int != 0 else 0

    # 3. Shape Analysis (Sphericity)
    try:
        verts, faces, normals, values = measure.marching_cubes(tumor_mask, spacing=spacing)
        surface_area = measure.mesh_surface_area(verts, faces)
        
        # Sphericity formula: ratio of sphere surface to actual surface
        sphericity = (np.pi**(1/3) * (6 * volume_mm3)**(2/3)) / surface_area
        sphericity = min(1.0, sphericity) # Cap at 1.0 (perfect sphere)
    except:
        surface_area = 0
        sphericity = 0

    return {
        "Volume (mm3)": round(volume_mm3, 2),
        "Surface Area (mm2)": round(surface_area, 2),
        "Sphericity (0-1)": round(sphericity, 3),
        "Mean Intensity": round(float(mean_int), 4),
        "Heterogeneity": round(float(heterogeneity), 4)
    }