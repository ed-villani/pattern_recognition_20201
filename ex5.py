from datetime import datetime
from decimal import PDF

from sklearn.model_selection import train_test_split

from commons.classifiers import get_data_for_classification, bayesian_classifier
from commons.commons import calculate_accuracy_percentage, save_to_csv, read_spambase
from commons.pdf import PDF, PDFTypes


def main():
    K = 10
    train_size = [0.9]
    accuracies_list = []
    for train in train_size:
        data = read_spambase()
        for i in range(K):
            try:
                X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], train_size=train)
                X_train_per_class = get_data_for_classification(y_train, X_train)
                classification = bayesian_classifier(
                    X_test, X_train_per_class, GaussianPDF(PDFTypes.MULTI_VAR)
                )
                accuracy = calculate_accuracy_percentage(y_test, classification)
                accuracies_list.append([i, round(accuracy, 4), train, datetime.now().strftime("%Y/%m/%dT%H:%M:%S")])
            except PDF:
                pass
    save_to_csv('/Users/eduardovillani/git/pattern_recognition_20201/data/ex_5.csv', accuracies_list)


if __name__ == '__main__':
    main()
