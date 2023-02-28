import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

data_path = "datasets/bird-detector-small/dataset/instances/Australian Pelican_aves_adult_pelecaniformes_pelecanidae_pelecanus_conspicillatus"



x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
z = [5, 5, 5, 5, 5, 9, 9, 9, 9, 9]


# plt.scatter(x, y)
# plt.show()

data = list(zip(x, y, z))
# inertias = []

# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data)
#     inertias.append(kmeans.inertia_)

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()