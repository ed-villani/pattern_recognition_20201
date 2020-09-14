from numpy.ma import arange

from random_data import RandomData, RandomDataPlotter, pdf


def main():
    # d1 = RandomData(100, 0.6, [2, 2], arange(0.06, 6, 0.06))
    # d2 = RandomData(100, 0.6, [4, 4], arange(0.06, 6, 0.06))

    # RandomDataPlotter.scatter_plot([d2, d1])
    # RandomDataPlotter.surface_plot([d2, d1])
    # RandomDataPlotter.contour_plot([d2, d1])
    x = pdf(
        s1=1.2, s2=1.2, p=0, x=1.5, u1=2.8, y=1.6, u2=2.9
    )
    print(x)


if __name__ == '__main__':
    main()
