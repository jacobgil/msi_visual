import cv2
import sys
import numpy as np
import random
from msi_visual.metrics import MSIVisualizationMetrics
from msi_visual.normalization import total_ion_count


src = cv2.imread(sys.argv[1])[:, :, ::-1]
dst = cv2.imread(sys.argv[2])[:, :, ::-1]
img = total_ion_count(np.load(sys.argv[3]))

res = cv2.seamlessClone(
    dst,
    src,
    None,
    (src.shape[1] // 2,
     src.shape[0] // 2),
    cv2.NORMAL_CLONE)

#res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)

#res = np.float32(src) + np.float32(dst)
#res = np.uint8(res/2)
#res = np.minimum(src, dst)

cv2.imshow("res", res[:, :, ::-1])
cv2.waitKey(-1)
random.seed(0)
metrics = MSIVisualizationMetrics(img, res, 3000).get_metrics()
print(metrics)