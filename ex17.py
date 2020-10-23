from copy import deepcopy

import numpy as np
from sklearn import svm

from commons.commons import calculate_accuracy_percentage
from commons.filters import Filters
from commons.plotter import print_figs
from commons.x_and_c import XAndC


def layer(data, conv_filter):
    aux_data = deepcopy(data)
    aux_data = Filters.convolution(aux_data, conv_filter)
    aux_data = Filters.relu(aux_data)
    aux_data = Filters.max_polling(aux_data, 2, 2)
    return np.reshape(aux_data, (1, np.product(aux_data.shape)))


def cnn(data, filter_list):
    cnn_list = np.array([layer(data, fi) for fi in filter_list])
    x = np.reshape(cnn_list, (1, cnn_list.shape[0] * cnn_list.shape[2]))

    # print_figs(np.reshape(x, (16, 4)))
    return x


def main():
    data, classification = XAndC()
    test_imgs = -1
    # for d in data:
    #     print_figs(np.reshape(d, (9, 9)))

    filter_list = XAndC.filters()

    # for f in filter_list:
    #     print_figs(f)

    data = [cnn(np.reshape(img, (9, 9)), filter_list) for img in data]
    data = np.reshape(np.array(data), (np.array(data).shape[0], np.array(data).shape[2]))

    X_train, X_test, y_train, y_test = data[:test_imgs], data[test_imgs:], classification[:test_imgs], classification[test_imgs:]

    svm_linear = svm.SVC(kernel='rbf')
    svm_linear.fit(X_train, y_train)
    try:
        classification = svm_linear.predict(X_test)
    except Exception:
        classification = svm_linear.predict([X_test])
    calculate_accuracy_percentage(y_test, classification)


if __name__ == '__main__':
    main()
