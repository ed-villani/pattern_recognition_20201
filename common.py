import numpy as np


def combine_pdf(pdfs_to_combine, weights=None):
    weights = np.ones(len(pdfs_to_combine)) if weights is None else weights

    def combined(*x):
        acc = 0
        for pdf, weight in zip(pdfs_to_combine, weights):
            acc += weight * pdf(*x)
        return acc

    return combined


def gaussian_mixture(total_gaussians, *train_classes):
    classes_data = []
    all_pdfs = []
    total_samples = sum([len(train_class) for train_class in train_classes])
    for train_class in train_classes:
        pdfs = []
        weights = []
        n_clusters = max(1, int(total_gaussians * (len(train_class) / total_samples)))
        clusters, centers, _ = cmn.k_means(n_clusters, train_class, n_clusters * 100)
        for cluster in clusters:
            cluster = np.array(cluster)
            pdf = cmn.pdf_from_data(cluster)
            pdfs.append(pdf)
            weights.append(cluster.shape[0] / total_samples)
        combined = combine_pdf(pdfs, weights)
        all_pdfs.append(combined)
        classes_data.append((clusters, centers))
    return classes_data, all_pdfs


def generate_pdfs_classifier(*pdfs):
    def classifier(*x):
        result = np.array([pdf(*x) for pdf in pdfs])
        return np.argmax(result)

    return classifier


def generate_gaussian_mixture_classifier(total_gaussians, *train_classes):
    classes_data, all_pdfs = gaussian_mixture(total_gaussians, *train_classes)
    return {
        "classifier": generate_pdfs_classifier(*all_pdfs),
        "side_effects": (classes_data, all_pdfs),
    }


def k_fold_cross_validate(
        n_folds,
        classifier_from_data_fn,
        *classes,
        side_effects_key=None,
        classifier_key="classifier",
):
    class_fold_samples = []
    accuracies = []
    classifiers = []
    side_effects = []
    for c in classes:
        assert c.shape[0] % n_folds == 0
        np.random.shuffle(c)
        class_fold_samples.append(int(c.shape[0] / n_folds))

    for i in range(n_folds):
        train_samples = []
        test_samples = []
        for c, fold_samples in zip(classes, class_fold_samples):
            c_train_1 = c[0: i * fold_samples]
            c_test = c[i * fold_samples: (i + 1) * fold_samples]
            c_train_2 = c[(i + 1) * fold_samples:]
            train_samples.append(np.vstack((c_train_1, c_train_2)))
            test_samples.append(c_test)
        result = classifier_from_data_fn(*train_samples)
        classifier = result[classifier_key]
        side_effect = result[side_effects_key] if side_effects_key is not None else None
        acc = cmn.get_accuracy(classifier, *test_samples)
        accuracies.append(acc)
        classifiers.append(classifier)
        side_effects.append(side_effect)
    best_index = np.array(accuracies).argmax()
    return accuracies, classifiers[best_index], side_effects[best_index]
