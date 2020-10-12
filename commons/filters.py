from dataclasses import dataclass

import numpy as np


@dataclass
class FilterType:
    SHARPEN = 'sharpen'
    HORIZONTAL_LINE = 'horizontal_line'
    VERTICAL_LINE = 'vertical_line'
    BORDER = 'border'


class Filters:
    def __new__(cls, filter_type):
        filter_type = filter_type.lower()
        if filter_type == 'sharpen':
            return np.reshape(np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]), (3, 3))
        elif filter_type == 'horizontal_line':
            return np.reshape(np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]), (3, 3))
        elif filter_type == 'vertical_line':
            return np.reshape(np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]), (3, 3))
        elif filter_type == 'border':
            return np.reshape(np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]), (3, 3))

    @staticmethod
    def convolution(img, img_filter):
        def get_img_portion(i, j, m):
            return img[i:i + m, j:j + m]

        def convolution_pixel_value(i, j, m):
            return np.sum(get_img_portion(i, j, m) * img_filter)

        def new_img_shape():
            m, n = img_filter.shape
            y, x = img.shape
            y = y - m + 1
            x = x - m + 1
            return x, y, m

        x, y, m = new_img_shape()
        new_image = [convolution_pixel_value(i, j, m) for i in range(y) for j in range(x)]
        return np.reshape(new_image, (x, y))