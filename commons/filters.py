from copy import deepcopy
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
    def new_img_shape(img, img_filter):
        m, n = img_filter.shape
        y, x = img.shape
        y = y - m + 1
        x = x - m + 1
        return x, y, m

    @staticmethod
    def get_img_portion(img, i, j, m):
        return img[i:i + m, j:j + m]

    @staticmethod
    def convolution(img, img_filter, function=None, **kwargs):

        def convolution_pixel_value(i, j, m):
            return np.sum(Filters.get_img_portion(img, i, j, m) * img_filter)

        x, y, m = Filters.new_img_shape(img, img_filter)
        new_image = [convolution_pixel_value(i, j, m) for i in range(y) for j in range(x)]
        new_image = np.reshape(new_image, (x, y)) / 9
        if function == 'ReLU':
            new_image = Filters.relu(new_image)
        elif function == 'max_polling':
            new_image = Filters.max_polling(new_image, **kwargs)
        return new_image

    @staticmethod
    def relu(array):
        return array.clip(min=0)

    @staticmethod
    def max_polling(img, px, py):
        aux_img = deepcopy(img)
        x, y = aux_img.shape
        if x % px:
            for _ in range(px - (x % px)):
                aux_img = np.vstack((aux_img, np.full((aux_img.shape[1]), -np.inf)))
        if y % py:
            for _ in range(px - (y % py)):
                aux_img = np.insert(aux_img, aux_img.shape[1], np.full(aux_img.shape[0], -np.inf), 1)

        x, y = tuple([int(shape / divisor) for shape, divisor in zip(aux_img.shape, [px, py])])
        new_image = [np.max(Filters.get_img_portion(aux_img, i * px, j * py, px)) for i in range(y) for j in range(x)]
        return np.reshape(new_image, (x, y))
