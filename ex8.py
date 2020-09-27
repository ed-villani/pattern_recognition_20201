from commons.classifiers import get_data_for_classification
from commons.commons import gen_data, join_data
from commons.plotter import scatter_plot
from commons.ward import ward


def main():
    Ks = [2, 4, 8]
    sigmas = 0.3
    mus = [
        [2, 2],
        [4, 4],
        [2, 4],
        [4, 2]
    ]
    N = (100, 2)
    data = [gen_data(mu, sigmas, N) for mu in mus]
    scatter_plot(data)
    for K in Ks:
        points = join_data(data)
        classes, classification, centers = ward(K, points)
        data_classified = get_data_for_classification(classification, points)
        scatter_plot(
            data_classified,
            centers=centers,
            title=f'Clusters for K = {K} and sigma = {sigmas}'
        )


if __name__ == '__main__':
    main()
