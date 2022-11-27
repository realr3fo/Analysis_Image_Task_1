import cv2
import numpy as np
from matplotlib import pyplot as plt


def binarize_by_adaptive(image: np.ndarray, block_size, constant, blur) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, blur, cv2.BORDER_DEFAULT)
    blurred = blurred.astype(np.uint8)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        constant,
    )
    return thresh


fig, axs = plt.subplots(2, 2)

original_img = cv2.imread("./dataset/1660060800.jpg", cv2.IMREAD_COLOR)
img_bg = cv2.imread("./dataset/1660021200.jpg", cv2.IMREAD_COLOR)

# resized_img = cv2.resize(original_img, (0, 0), fx=0.8, fy=0.8)
resized_img = original_img.copy()
roi = resized_img[450:, :]

resized_img_bg = img_bg.copy()
roi_bg = resized_img_bg[450:, :]
subtracted = cv2.subtract(roi_bg, roi)

# remove ocean
lower_ocean = np.array([0, 10, 20])
upper_ocean = np.array([60, 80, 100])
ocean_mask = cv2.inRange(subtracted, lower_ocean, upper_ocean)
res = cv2.bitwise_and(subtracted, subtracted, mask=ocean_mask)
subtracted = cv2.subtract(subtracted, res)

img_gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
thresh = binarize_by_adaptive(img_gray, 21, 5, (5, 5))

kernel1 = np.ones((4, 4), dtype=np.uint8)
kernel2 = np.ones((3, 3), dtype=np.uint8)
morphed = thresh
morphed = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel1)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_DILATE, kernel2)

# process upper right
roi_copy = roi.copy()
upper_right = roi_copy[:225, 1285:]
upper_right_bg = roi_bg[:225, 1285:]
upper_right_subtracted = cv2.subtract(upper_right_bg, upper_right)

# remove shadow
lower_shadow = np.array([0, 10, 10])
upper_shadow = np.array([80, 85, 100])
upper_right_shadow_mask = cv2.inRange(
    upper_right_subtracted, lower_shadow, upper_shadow
)
res = cv2.bitwise_and(
    upper_right_subtracted, upper_right_subtracted, mask=upper_right_shadow_mask
)
upper_right_subtracted = cv2.subtract(upper_right_subtracted, res)
upper_img_gray = cv2.cvtColor(upper_right_subtracted, cv2.COLOR_BGR2GRAY)
upper_right_thresh = binarize_by_adaptive(upper_img_gray, 15, 1, (5, 5))
kernel3 = np.ones((4, 4), dtype=np.uint8)
kernel4 = np.ones((10, 10), dtype=np.uint8)
upper_morphed = upper_right_thresh
upper_morphed = cv2.morphologyEx(upper_morphed, cv2.MORPH_ERODE, kernel3)
upper_contours, _ = cv2.findContours(
    upper_morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)


boxes = []
for cnt in upper_contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if y > 100:
        if 425 > area > 5:
            x, y, w, h = cv2.boundingRect(cnt)
            x = x + 1295
            boxes.append([x, y, w, h])
    else:
        if 100 > area > 5:
            x, y, w, h = cv2.boundingRect(cnt)
            x = x + 1295
            boxes.append([x, y, w, h])

contours, _ = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 900 > area:
        constant = -10
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 1285:
            boxes.append([x, y, w, h])
        else:
            if y > 225:
                boxes.append([x, y, w, h])
for box in boxes:
    x, y, w, h = box[0], box[1], box[2], box[3]
    roi_square = roi[y : y + h, x : x + w]
    cv2.rectangle(roi, (x + constant, y), (x + w + constant, y + h), (200, 0, 0), 1)

axs[0, 0].imshow(upper_morphed, cmap="gray")
axs[0, 1].imshow(morphed, cmap="gray")
axs[1, 0].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
axs[1, 1].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.show()
