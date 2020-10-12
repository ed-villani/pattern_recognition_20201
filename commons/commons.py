import ssl
from random import choice, randint

import numpy as np
import pandas as pd
from numpy.random.mtrand import shuffle, normal
from sklearn import datasets


def random_colors(number_of_colors):
    return ["#" + ''.join([choice('0123456789ABCDEF') for _ in range(6)])
            for _ in range(number_of_colors)]


def save_to_csv(filename, data):
    df = pd.DataFrame(data)
    # csv_data = df.to_csv(index=False)
    df.to_csv(filename, index=False, mode='a+', header=False)
    print(f'Saving Data to CSV, file {filename}')


def read_spambase():
    datContent = [i.strip().split(',') for i in open("data/spambase.data").readlines()]
    for dat in datContent:
        for index, i in enumerate(dat):
            dat[index] = float(i)
    datContent = np.array(datContent)
    return datContent


def read_heart():
    datContent = [i.strip().split() for i in open("data/heart.dat").readlines()]
    for dat in datContent:
        for index, i in enumerate(dat):
            dat[index] = float(i)
    datContent = np.array(datContent)
    datContent[:, -1] = datContent[:, -1] - 1
    return datContent


def join_classification(X_train, y_train, X_test, classification):
    data = np.concatenate((X_train, X_test))
    classification = np.concatenate((y_train, classification))
    return classification, data


def join_data(data, shuffle=True):
    if shuffle:
        aux = np.concatenate([d.T for d in data])
        np.random.mtrand.shuffle(aux)
        return aux
    return np.concatenate([d.T for d in data])


def gen_data(mu, sigma, N, classifiation=None):
    if classifiation is None:
        return normal(mu, sigma, size=N).T
    data = normal(mu, sigma, size=N).T

    return np.insert(data, len(data), np.array([classifiation for _ in data.T]), 0)


def calculate_accuracy_percentage(y_test, classification):
    if len(y_test) != len(classification):
        raise Exception
    hit = 0
    for p, c in zip(y_test, classification):
        if p != c:
            hit += 1
    accuracy = 1 - (hit / len(classification))
    print(f'We got an accuracy of {round(accuracy * 100, 2)}%')
    return accuracy


def get_random_face():
    ssl._create_default_https_context = ssl._create_unverified_context
    faces = datasets.fetch_olivetti_faces().data
    return np.reshape(faces[randint(0, len(faces) - 1)], (64, 64))
