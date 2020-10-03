import ssl
from datetime import datetime

from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from commons.classifiers import bayesian_classifier, get_data_for_classification
from commons.commons import calculate_accuracy_percentage, save_to_csv
from commons.pdf import PDF, PDFTypes
from commons.plotter import confusion_matrix


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    faces = datasets.fetch_olivetti_faces()
    # print_figs(faces)
    points, classification = faces.data, faces.target
    total_n_components = 21
    n_components= 21
    h = 0.6
    train_size = 0.5

    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(points.T)
    transformed_points = pca.fit_transform(points)
    better_accuracy = 0
    accuracies_list = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(transformed_points, classification, train_size=train_size)
        data_classified = get_data_for_classification(y_train, X_train, n_classes=len(set(classification)))

        bayesian_classification = bayesian_classifier(
            X_test,
            data_classified,
            PDF(PDFTypes.KDE),
            h=h
        )

        accuracy = calculate_accuracy_percentage(y_test, bayesian_classification)
        accuracies_list.append(
            [i, round(accuracy, 4), 0.9, h, n_components, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])

        if accuracy > better_accuracy:
            better_classification = bayesian_classification
            better_X_test = X_test
            better_y_test = y_test

    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_13.csv', accuracies_list)
    confusion_matrix(
        metrics.confusion_matrix(better_y_test, better_classification)
    )


if __name__ == '__main__':
    main()
