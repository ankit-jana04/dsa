import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
#turn the data into a set of points
data=list(zip(x,y))
print(data)


#Kmeans
from sklearn.cluster import KMeans
inertias = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range (1,11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#We can see that the “elbow” on the graph above (where the inertia become more
#linear) is at k-2, We can then fit our K-means algorithm one more time and plot the
#different clusters assigned to the data:
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
plt.scatter(x, y, c=kmeans.labels_)
plt.show()