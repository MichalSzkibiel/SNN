import numpy as np


def euclidean_distance(a, b):
    return np.sum((a - b)**2, 1)**0.5


def search_insert(arr, value, k):
    begin_idx = 0
    end_idx = len(arr)
    if arr[end_idx - 1, 1] <= value[1]:
        return (arr + [value])[:k]
    while begin_idx < end_idx:
        cur_idx = (begin_idx + end_idx) // 2
        if arr[cur_idx][1] <= value[1]:
            begin_idx = cur_idx + 1
        else:
            end_idx = cur_idx
    return arr[:begin_idx] + [value] + arr[begin_idx:k - 1]


def knn_analysis(dataset, k, metric=euclidean_distance):
    n = dataset.shape[0]
    d = [[] for i in range(n)]
    for i in range(1, n//2):
        dists = metric(dataset, np.block([[dataset[i:]], [dataset[:i]]]))
        for j in range(n):
            if i == n/2 - 1 and j + i >= n:
                break
            l = (j + i) % n
            d[j] = search_insert(d[j], [l, dists[j]], k)
            d[l] = search_insert(d[l], [j, dists[j]], k)
    return [{el2[0] for el2 in el1} for el1 in d]


def snn(dataset, k, k_t, metric=euclidean_distance):
    knn = knn_analysis(dataset, k, metric)
    labels = [i for i in range(len(knn))]
    for i in range(len(knn)):
        for j in knn[i]:
            if i in knn[j] and len(knn[i].intersection(knn[j])) >= k_t:
                labels[j] = labels[i]
    return labels
