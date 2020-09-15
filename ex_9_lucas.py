# %%
import numpy as np
import matplotlib.pyplot as plt
import common as cmn
import tabulate
import pandas as pd


def generate_kde_pdf(data, spread=None):
    total_samples = data.shape[0]
    dimensions = data.shape[1]
    if spread is None:
        spread = max(1.06 * data.std() * total_samples ** (-1 / 5), 0.00001)
    multiplier = 1 / (total_samples * (((2 * np.pi) * (1 / 2) * spread) * dimensions))

    def pdf(*x):
        x_array = np.array(x)
        all_terms = np.e * -(((x_array - data) * 2) / (2 * spread ** 2))
        return multiplier * all_terms.prod(axis=1).sum()

    return pdf


def generate_kde_classifier(*classes, spread=None):
    all_pdfs = []
    for samples in classes:
        all_pdfs.append(generate_kde_pdf(samples, spread))
    kde_classifier = cmn.generate_pdfs_classifier(*all_pdfs)
    return {"classifier": kde_classifier, "side_effects": all_pdfs}


def main():
    # %%


    all, class_indicator = cmn.twospirals(1000, 1, 1.1)

    plt.title("Base de dados:")
    plt.plot(
        all[class_indicator == 0, 0], all[class_indicator == 0, 1], ".", label="classe 1"
    )
    plt.plot(
        all[class_indicator == 1, 0], all[class_indicator == 1, 1], ".", label="classe 2"
    )
    plt.legend()
    plt.show()

    # Dividindo as classes
    c1 = all[class_indicator == 0, :]
    c2 = all[class_indicator == 1, :]

    # %%
    # %%
    # Definido parâmetros 'h' a serem testados
    h_base = max(1.06 * all.std() * all.shape[0] ** (-1 / 5), 0.00001)
    n_tests = 9
    step = h_base / (n_tests - 1)
    all_h = [(h_base * 1.5) - (step * i) for i in range(n_tests)]

    h_accuracies = []
    best_classifier = None
    best_accuracies = None
    best_pdfs = None
    for h in all_h:
        accuracies, classifier, pdfs = cmn.k_fold_cross_validate(
            10,
            lambda *classes: generate_kde_classifier(*classes, spread=h),
            c1,
            c2,
            side_effects_key="side_effects",
        )
        accuracy = accuracies.mean()
        if best_classifier is None or accuracy > max(h_accuracies):
            print(f"New Best is {h} with {accuracy}")
            best_classifier = classifier
            best_pdfs = pdfs
            best_accuracies = accuracies
            best_h = h
        else:
            print(f"Failed {h} with {accuracy}")
        h_accuracies.append(accuracy)

    # %%

    tabulate.tabulate([h_accuracies], tuple(f"h={h}" for h in all_h), tablefmt='html')

    # %%

    x_range = y_range = np.linspace(-7, 7, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z1 = cmn.get_z_grid(best_pdfs[0], X, Y)
    Z2 = cmn.get_z_grid(best_pdfs[1], X, Y)
    fig = plt.figure()
    cmn.contour_plot(fig.add_subplot(1, 1, 1), X, Y, Z1, Z2)
    plt.show()


if __name__ == '__main__':
    main()
