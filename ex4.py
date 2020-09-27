from datetime import datetime

from sklearn.model_selection import train_test_split

from commons.classifiers import bayesian_classifier, get_data_for_classification
from commons.commons import read_heart, calculate_accuracy_percentage, save_to_csv
from commons.pdf import GaussianPDF, GaussianPDFTypes


def main():
    K = 10
    train_size = [0.9, 0.7, 0.2]
    accuracies_list = []
    for train in train_size:
        data = read_heart()
        for i in range(K):
            X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], train_size=train)
            X_train_per_class = get_data_for_classification(y_train, X_train)
            classification = bayesian_classifier(
                X_test, X_train_per_class, GaussianPDF(GaussianPDFTypes.MULTI_VAR)
            )
            accuracy = calculate_accuracy_percentage(y_test, classification)
            accuracies_list.append([i, round(accuracy, 4), train, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])
    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_4.csv', accuracies_list)


if __name__ == '__main__':
    main()
