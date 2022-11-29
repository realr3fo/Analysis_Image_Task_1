import cv2
import numpy as np
from matplotlib import pyplot as plt


# Function to binarize image by adaptive Threshold
def binarize_image_by_adaptive(
    image: np.ndarray, block_size, constant, blur
) -> np.ndarray:
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


# Function to find certain region from image by pixel ranges
def find_region_by_pixel(
    image: np.ndarray, lower_bgr_limit: np.ndarray, upper_bgr_limit: np.ndarray
) -> np.ndarray:
    ocean_mask = cv2.inRange(image, lower_bgr_limit, upper_bgr_limit)
    result = cv2.bitwise_and(image, image, mask=ocean_mask)
    return result


# Main code to find person from an image
def find_person_from_beach_image(file_name: str, show_plot: bool = True):
    # File load and read
    bg_file_name = "./dataset/1660021200.jpg"
    original_img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img_bg = cv2.imread(bg_file_name, cv2.IMREAD_COLOR)

    # Region of interest
    image_copy = original_img.copy()
    roi = image_copy[450:, :]
    img_bg_copy = img_bg.copy()
    roi_bg = img_bg_copy[450:, :]

    # Removing background from roi
    roi_fg = cv2.subtract(roi_bg, roi)

    # Remove ocean
    lower_ocean_bgr_limit = np.array([0, 10, 20])
    upper_ocean_bgr_limit = np.array([60, 80, 100])
    roi_with_ocean = find_region_by_pixel(
        roi_fg, lower_ocean_bgr_limit, upper_ocean_bgr_limit
    )
    roi_fg = cv2.subtract(roi_fg, roi_with_ocean)

    # Apply binarization to roi foreground
    roi_gray = cv2.cvtColor(roi_fg, cv2.COLOR_BGR2GRAY)
    roi_thresh = binarize_image_by_adaptive(roi_gray, 21, 5, (5, 5))

    # Apply morphology erotion and dilation to binarized roi
    KERNEL_3 = np.ones((3, 3), dtype=np.uint8)
    KERNEL_4 = np.ones((4, 4), dtype=np.uint8)
    roi_morphed = cv2.morphologyEx(roi_thresh, cv2.MORPH_ERODE, KERNEL_4)
    roi_processed = cv2.morphologyEx(roi_morphed, cv2.MORPH_DILATE, KERNEL_3)

    # Process lifeguard region the same way as previous one,
    # but with different configuration
    roi_copy = roi.copy()
    lifeguard = roi_copy[:225, 1285:]
    lifeguard_bg = roi_bg[:225, 1285:]
    lifeguard_fg = cv2.subtract(lifeguard_bg, lifeguard)

    lower_shadow = np.array([0, 10, 10])
    upper_shadow = np.array([80, 85, 100])
    lifeguard_roi_with_shadow = find_region_by_pixel(
        lifeguard_fg, lower_shadow, upper_shadow
    )
    lifeguard_fg = cv2.subtract(lifeguard_fg, lifeguard_roi_with_shadow)
    lifeguard_gray = cv2.cvtColor(lifeguard_fg, cv2.COLOR_BGR2GRAY)
    lifeguard_thresh = binarize_image_by_adaptive(lifeguard_gray, 15, 1, (5, 5))
    lifeguard_morphed = cv2.morphologyEx(lifeguard_thresh, cv2.MORPH_ERODE, KERNEL_4)

    # Boxes array will save the boundary boxes
    boxes = []
    lifeguard_contours, _ = cv2.findContours(
        lifeguard_morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in lifeguard_contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        y += 450
        x += 1295
        if y > 550:
            if 425 > area > 5:
                boxes.append([x, y, w, h])
        else:
            if 100 > area > 5:
                boxes.append([x, y, w, h])

    contours, _ = cv2.findContours(
        roi_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        x -= 10
        y += 450
        if 900 > area:
            if x < 1285:
                boxes.append([x, y, w, h])
            else:
                if y > 650:
                    boxes.append([x, y, w, h])
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (200, 0, 0), 1)
    if show_plot:
        _, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        title = "Person detection for " + file_name
        plt.title(title)
        plt.gcf().set_dpi(150)
        plt.show()
    return boxes


if __name__ == "__main__":
    file_name = "./dataset/1660050000.jpg"
    find_person_from_beach_image(file_name, True)
