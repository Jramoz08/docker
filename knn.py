import math
from collections import Counter

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        distances = []
        for i in range(len(self.X_train)):
            distance = euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))
        
        distances.sort(key=lambda dist: dist[0])
        k_nearest_labels = [dist[1] for dist in distances[:self.k]]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def find_nearest_points(self, x):
        distances = []
        for i in range(len(self.X_train)):
            distance = euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.X_train[i]))
        
        distances.sort(key=lambda dist: dist[0])
        nearest_points = [dist[1] for dist in distances[:self.k]]
        return nearest_points

# Datos de entrenamiento
X = [
    [2, 3],
    [7, 5],
    [6, 5],
    [7, 5],
    [7, 3],
    [8, 2],
    [11, 15],
    [13, 12],
    [14, 13],
    [17, 19]
]
y = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  

knn = KNN(k=3)
knn.fit(X, y)

# Punto nuevo fijo
new_point = [6, 6]

nearest_points = knn.find_nearest_points(new_point)
print(f"Puntos más cercanos a {new_point}: {nearest_points}")

