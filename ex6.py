from commons.classifiers import get_data_for_classification
from commons.commons import gen_data, join_data

from commons.fkm import FuzzyKMeans
from commons.plotter import scatter_plot


def main():
    Ks = [2, 4, 8]
    sigmas = [0.3, 0.5, 0.7]
    mus = [
        [2, 2],
        [4, 4],
        [2, 4],
        [4, 2]
    ]
    N = (100, 2)

    for sigma in sigmas:
        data = [gen_data(mu, sigma, N) for mu in mus]
        for K in Ks:
            points = join_data(data)
            scatter_plot(data)
            fkm = FuzzyKMeans(points, K, 1e-19).fkm()
            classification = FuzzyKMeans.normalize_membership_matrix(fkm[0])
            data_classified = get_data_for_classification(classification, points)
            scatter_plot(
                data_classified,
                centers=fkm[1],
                title=f'Clusters for K = {K} and sigma = {sigma}'
            )


if __name__ == '__main__':
    main()
