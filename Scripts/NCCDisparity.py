
import numpy as np
import cv2

max_disparity = 48
block_size = 23
gray_img_l = cv2.imread('left1.jpg', cv2.IMREAD_GRAYSCALE)
gray_img_r = cv2.imread('right1.jpg', cv2.IMREAD_GRAYSCALE)
disparity_map = np.zeros_like(gray_img_l, dtype='float32')

padding = block_size // 2

for row_l in range(padding, gray_img_l.shape[0]-padding):
    for col_l in range(padding, gray_img_l.shape[1]-padding):
        template_l = gray_img_l[row_l-padding:row_l+padding+1, col_l-padding:col_l+padding+1]

        best_ncc = -np.inf
        best_disparity = 0

        for d in range(max_disparity+1):
            col_r = col_l - d

            if col_r >= padding and col_r <= gray_img_r.shape[1]-padding-1:
                template_r = gray_img_r[row_l-padding:row_l+padding+1, col_r-padding:col_r+padding+1]
                numerator = np.sum((template_l - np.mean(template_l)) * (template_r - np.mean(template_r)))
                denominator = np.sqrt(np.sum((template_l - np.mean(template_l))*2)) * np.sqrt(np.sum((template_r - np.mean(template_r))*2))
                NCC = numerator / denominator
                if NCC > best_ncc:
                    best_ncc = NCC
                    best_disparity = d

        disparity_map[row_l,col_l] = best_disparity

disparity_map /= np.max(disparity_map)
disparity_map = (disparity_map * 255).astype('uint8')

cv2.imshow('Disparity Map', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()