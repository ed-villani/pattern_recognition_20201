from commons.commons import get_random_face
from commons.filters import Filters, FilterType
from commons.plotter import print_figs


def main():
    face = get_random_face()
    print_figs(face, title=f'Original Face')
    filter_list = [f for f in FilterType.__dict__ if '__' not in f]
    for fi in filter_list:
        f = Filters(fi)
        new_img = Filters.convolution(face, f)
        print_figs(new_img, title=f'Face {fi}')


if __name__ == '__main__':
    main()
