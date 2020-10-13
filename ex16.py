from commons.commons import get_random_face
from commons.filters import Filters, FilterType
from commons.plotter import print_figs


def main():
    face = get_random_face()
    print_figs(face, title=f'Original Face')
    f = Filters(FilterType.BORDER)

    new_img_no_relu = Filters.convolution(face, f)
    print_figs(new_img_no_relu, title=f'Only Filter Face {FilterType.BORDER}')

    new_img_relu = Filters.convolution(face, f, function="ReLU")
    print_figs(new_img_relu, title=f'ReLU Face {FilterType.BORDER}')

    new_img = Filters.convolution(face, f, function="max_polling", px=2, py=2)
    print_figs(new_img, title=f'Max Polling Face {FilterType.BORDER}')


if __name__ == '__main__':
    main()
