import ssl
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from commons.classifiers import bayesian_classifier, fuzzy_class_by_inner_class, get_data_for_classification
from commons.commons import save_to_csv, calculate_accuracy_percentage
from commons.fkm import FuzzyKMeans
from commons.pdf import PDF, PDFTypes


def main():
    ssl._create_default_https_context = ssl._create_unverified_context
    K = 30
    n_splits = 10
    data = load_breast_cancer()
    points, classification = data.data, data.target
    accuracies_list = []
    i = 0
    for train_index, test_index in KFold(n_splits=n_splits).split(points):
        points_train = points[train_index]
        class_train = classification[train_index]

        points_test = points[test_index]
        class_test = classification[test_index]

        data_classified = get_data_for_classification(class_train, points_train)
        fkm = [FuzzyKMeans(c.T, int(K / 2), 1e-19).fkm() for c in data_classified]

        fuzzy_data_classified = fuzzy_class_by_inner_class(fkm, data_classified)
        bayesian_classification = bayesian_classifier(
            points_test,
            fuzzy_data_classified,
            PDF(PDFTypes.MIXTURE),
            'mix'
        )
        accuracy = calculate_accuracy_percentage(class_test, bayesian_classification)
        accuracies_list.append([i, round(accuracy, 4), 0.9, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])

        i += 1

    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_9_2.csv', accuracies_list)


if __name__ == '__main__':
    main()
