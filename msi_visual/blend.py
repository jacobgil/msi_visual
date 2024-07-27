import cv2
import sys

src = cv2.imread(sys.argv[1])
dst = cv2.imread(sys.argv[2])

res = cv2.seamlessClone(
    src,
    dst,
    None,
    (src.shape[1] // 2,
     src.shape[0] // 2),
    cv2.MIXED_CLONE)
cv2.imwrite(sys.argv[3], res)
