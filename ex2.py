from numpy.ma import arange

from random_data import RandomData, RandomDataPlotter, data_frontier


def main():
    grid = arange(0.06, 6, 0.06)

    d1 = RandomData(100, 0.6, [2, 2], grid)
    d2 = RandomData(100, 0.8, [4, 4], grid)
    d3 = RandomData(100, 0.2, [2, 4], grid)
    d4 = RandomData(100, 1, [4, 2], grid)

    RandomDataPlotter.scatter_plot([d4, d3, d2, d1])
    RandomDataPlotter.surface_plot([d4, d3, d2, d1])
    # for d in [d4, d3, d2, d1]:
    #     RandomDataPlotter.surface_plot([d])
    RandomDataPlotter.contour_plot([d4, d3, d2, d1])
    frontier = {
        'grid_x': grid,
        'grid_y': grid,
        'frontier': data_frontier([d4, d3, d2, d1], grid)
    }
    RandomDataPlotter.scatter_plot([d4, d3, d2, d1], frontier)


if __name__ == '__main__':
    main()
