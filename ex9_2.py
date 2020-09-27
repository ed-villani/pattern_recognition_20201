# import numpy as np
# import pandas as pd
# from numpy.linalg import pinv, det
# from numpy.random.mtrand import shuffle
# from sklearn.model_selection import KFold
#
# from commons.fkm import FuzzyKMeans
#
#
# def pdf(n, K, x, m):
#     from decimal import Decimal
#     multiplier = 10 ** 16
#     divider = multiplier ** K.shape[0]
#     d = (1 / np.sqrt(Decimal(((2 * np.pi) ** n)) * Decimal(det(K * multiplier)) / Decimal(divider)))
#     e = Decimal(np.exp(-(0.5 * ((x - m) @ pinv(K)) @ (x - m))))
#     return float(d * e)


def main():
    pass
#     np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
#     datContent = [i.strip().split() for i in open("data/cancer.dat").readlines()]
#     data, classes = [], []
#     for index, dat in enumerate(datContent):
#         if index == 0:
#             continue
#         data.append([])
#         drop = False
#         for i, d in enumerate(dat):
#             if 1 < i < 11:
#                 if d == 'NA':
#                     drop = True
#                     d = -1
#                 data[len(data) - 1].append(int(d))
#             if d == 'benign':
#                 classes.append(1)
#             elif d == 'malignant':
#                 classes.append(2)
#         if drop:
#             data.pop(-1)
#             classes.pop(-1)
#     classes = np.array(classes).astype(np.int64)
#     data = np.array(data)
#     data = np.insert(data, 0, classes, axis=1)
#     shuffle(data)
#     data = data[:,1:]
#     classes = data[:,0]
#     for index, d in enumerate(data):
#         data[index] = np.array(d, dtype=np.float)
#     kf = KFold(n_splits=10)
#     K = 30
#     accuracy = []
#
#     better_accuracy = np.inf
#     for train_index, test_index in kf.split(data):
#         points_train = data[train_index]
#         class_train = classes[train_index]
#
#         points_train_1 = points_train[np.where(class_train == 1)]
#         class_train_1 = class_train[np.where(class_train == 1)]
#
#         points_train_2 = points_train[np.where(class_train == 2)]
#         class_train_2 = class_train[np.where(class_train == 2)]
#
#         points_test = data[test_index]
#         class_test = classes[test_index]
#
#         fkm_1 = FuzzyKMeans(points_train_1, int(K / 2), 1e-19).fkm()
#         fkm_2 = FuzzyKMeans(points_train_2, int(K / 2), 1e-19).fkm()
#         # fkm = FuzzyKMeans(points_train, K, 1e-19).fkm()
#         # scatter_plot(
#         #     points_train.T,
#         #     [colors[int(np.argmax(u)) - 1] for u in fkm[0]],
#         #     fkm[1].T,
#         #     f'spiral_k_{K}.png'
#         # )
#
#         point_class_1, points_in_classes_1 = pointers_per_classes(int(K / 2), fkm_1, points_train_1)
#         point_class_2, points_in_classes_2 = pointers_per_classes(int(K / 2), fkm_2, points_train_2)
#
#         # point_class, points_in_classes = pointers_per_classes(K, fkm, points_train)
#         # c_1_clusters, c_2_clusters = get_clussters_per_class(K, point_class, class_train)
#
#         p_x = len(points_train)
#         p_1 = len(np.where(class_train == 1)[0]) / p_x
#         p_2 = 1 - p_1
#         final_result = []
#         for p in points_test:
#             try:
#                 pdf_1 = sum([pdf(len(p), np.cov(c.T), p, np.mean(c.T, axis=1)) if len(c) else 0 for c in
#                          points_in_classes_1])
#             except Exception:
#                 i= 0
#             pdf_2 = sum([pdf(len(p), np.cov(c.T), p, np.mean(c.T, axis=1)) if len(c) else 0 for c in
#                          points_in_classes_2])
#
#             if (p_1 * pdf_1) / (p_2 * pdf_2) >= 1:
#                 final_result.append(1)
#             else:
#                 final_result.append(2)
#         hit = 0
#         for fr, r in zip(class_test, final_result):
#             if r != fr:
#                 # print(f"Result: {fr}, Actual Class: {r}")
#                 hit = hit + 1
#         # fold_accuracy = round((1 - abs(result * 2)) * 100, 2)
#         print(f"{i} Hit: {hit}")
#
#         if better_accuracy > hit:
#             print(f"Gotta a smaller hit: {hit}")
#             better_accuracy = hit
#             # better_points = np.concatenate((spiral.points[train_index], spiral.points[test_index]))
#             # better_classes = np.concatenate((spiral.classification[train_index], np.array(final_result)))
#             # better_points = spiral.points[test_index]
#             # better_classes = np.array(final_result)
#             # better_points_train = spiral.points[train_index]
#             # better_classes_train = spiral.classification[train_index]
#             # better_class_test = spiral.classification[test_index]
#             # better_points_test = spiral.points[test_index]
#         accuracy.append(hit)
#         i = i + 1
#     print(f"Mean: {np.mean(accuracy)}")
#     print(f"SD: {np.std(accuracy)}")
#
#
# def pointers_per_classes(K, fkm, data):
#     point_class = []
#     points_in_classes = [[] for _ in range(K)]
#     for index, point, in enumerate(data):
#         points_in_classes[int(np.argmax(fkm[0][index])) - 1].append(
#             point)
#         point_class.append(int(np.argmax(fkm[0][index])))
#
#     for index, c in enumerate(points_in_classes):
#         points_in_classes[index] = np.array(c)
#     return point_class, np.array(points_in_classes, dtype=object)
#
#
# def get_clussters_per_class(K, point_class, data):
#     class_1_k = []
#     class_2_k = []
#     df = pd.DataFrame(np.array([data, point_class]).T, columns=['class', 'cluster'])
#     df = df.groupby(['class', 'cluster']).size().reset_index()
#     for i in range(K):
#         if df[(df['cluster'] == i) & (df['class'] == 1)].empty:
#             class_2_k.append(i)
#         elif df[(df['cluster'] == i) & (df['class'] == 2)].empty:
#             class_1_k.append(i)
#         else:
#             class_1_val = df[(df['cluster'] == i) & (df['class'] == 1)].iloc[0][0]
#             class_2_Val = df[(df['cluster'] == i) & (df['class'] == 2)].iloc[0][0]
#
#             if class_1_val > class_2_Val:
#                 class_1_k.append(i)
#             else:
#                 class_2_k.append(i)
#     return class_1_k, class_2_k


if __name__ == '__main__':
    main()
