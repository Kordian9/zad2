import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def odczyt_pkt():
    with open(file="zadanie2.csv", newline='') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        for x, y, z in read:
            yield float(x), float(y), float(z)


if __name__ == '__main__':
    odczytane_pkt = list(odczyt_pkt())
    X, Y, Z = zip(*odczytane_pkt)
    a = np.array(odczytane_pkt)

    cluster = DBSCAN(eps=15, min_samples=10).fit(a)
    labels = cluster.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    n_clusters = 3
    k_means = KMeans(n_clusters=n_clusters)
    k_means = k_means.fit(a)
    labels = k_means.predict(a)

    yellow = labels == 0
    red = labels == 1
    black = labels == 2

    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(projection='3d')
    ax_2.scatter(a[yellow, 0], a[yellow, 1], a[yellow, 2], marker='o')
    ax_2.scatter(a[red, 0], a[red, 1], a[red, 2], marker='^')
    ax_2.scatter(a[black, 0], a[black, 1], a[black, 2], marker='x')
    plt.show()