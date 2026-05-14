import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

file_path = "/Users/merinphilip/Desktop/Main_project/proj_backend/AMOS22/imageTs/imaging2.nii.gz"
nii_img = nib.load(file_path)
img_data = nii_img.get_fdata()

# Initial slice
slice_index = img_data.shape[2] // 2

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

img_display = ax.imshow(img_data[:, :, slice_index], cmap="gray")
ax.set_title(f"Slice {slice_index}")
ax.axis("off")

# Slider setup
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, img_data.shape[2]-1,
                valinit=slice_index, valstep=1)

def update(val):
    slice_idx = int(slider.val)
    img_display.set_data(img_data[:, :, slice_idx])
    ax.set_title(f"Slice {slice_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
