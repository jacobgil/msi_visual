from msi_visual.parametric import ParametricModel
from msi_visual.kmeans_segmentation import KmeansSegmentation
from msi_visual.saliency_opt import SaliencyOptimization
import numpy as np
from msi_visual.normalization import total_ion_count, spatial_total_ion_count
from PIL import Image
import cv2

x = np.load(r"E:\MSImaging-data\_msi_visual\Extractions\NRL4506-5um\Intelli-slide\20_bins\0.npy")
#x = np.load(r"E:\MSImaging-data\_msi_visual\Extractions\NRL4509\s1-r2\5_bins\0.npy")
x = total_ion_count(x)
print("Loaded image", x.shape)

model = SaliencyOptimization(1000, 0.001)
model = ParametricModel(model, 8)

output = model(x)

print("output", output.shape)

output = cv2.merge([cv2.equalizeHist(output[:, :, i]) for i in range(3)])
output[x.max(axis=-1) == 0] = 0


Image.fromarray(output).save("NRL4506-5um_20bins_saliency_8downsample.png")