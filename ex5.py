from random import sample

from numpy import mean, array, cov, concatenate

from ex3_1_new import pdf_2


def main():
    percentage_training = 0.9
    datContent = [i.strip().split(',') for i in open("spambase.data").readlines()]
    for dat in datContent:
        for index, i in enumerate(dat):
            dat[index] = float(i)
    datContent = array(datContent)
    k = sample(range(len(datContent)), k=int(len(datContent) * percentage_training))
    data_traning = datContent[k].tolist()
    data = [d for index, d in enumerate(datContent.tolist()) if index not in k]
    data = array(data)
    data_traning = array(data_traning)

    c1_training = []
    c2_training = []
    for d in data_traning:
        if d[-1] == 1:
            c1_training.append(d[:-1])
        else:
            c2_training.append(d[:-1])

    c1_training = array(c1_training)
    c2_training = array(c2_training)

    c1_random = []
    c2_random = []
    for d in data:
        if d[-1] == 1:
            c1_random.append(d[:-1])
        else:
            c2_random.append(d[:-1])

    c1_random = array(c1_random)
    c2_random = array(c2_random)
    data_con = concatenate((c1_random, c2_random))
    pdf_c1 = [
        pdf_2(c1_training.shape[1], cov(c1_training.T), d, mean(c1_training.T, axis=1)) for d in data_con
    ]
    pdf_c2 = [
        pdf_2(c1_training.shape[1], cov(c2_training.T), d, mean(c2_training.T, axis=1)) for d in data_con
    ]

    total = concatenate((c1_training, c2_training)).shape[0]
    n_c1 = c1_training.shape[0] / total
    n_c2 = 1 - n_c1
    i = 0
    final_c1 = []
    final_c2 = []

    for x, y, d in zip(pdf_c1, pdf_c2, data_con):
        k = (x * n_c1) / (y * n_c2)
        if k < 1:
            final_c1.append(d)
        else:
            final_c2.append(d)
    final_c1 = array(final_c1)
    final_c2 = array(final_c2)


    print(f"Expected C1 (Spam): {c1_random.T.shape[1]} - Result C1: {final_c1.shape[0]}")
    print(f"Expected C2 (Not Spam): {c2_random.T.shape[1]} - Result C2: {final_c2.shape[0]}")
    print(f"Error: {abs(((final_c1.shape[0] - c1_random.T.shape[1]) / (c1_random.T.shape[1] + c2_random.T.shape[1])) * 100)}%\n")


def iterar():
    for i in range(1):
        print(i)
        main()


if __name__ == '__main__':
    iterar()
