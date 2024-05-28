import math
import pytest
from knn import KNN, euclidean_distance  # Asegúrate de que 'knn_example' es el nombre correcto del archivo principal

def test_euclidean_distance():
    point1 = [1, 2]
    point2 = [4, 6]
    assert euclidean_distance(point1, point2) == 5.0

def test_predict():
    X_train = [[1, 1], [2, 2], [3, 3]]
    y_train = [0, 0, 1]
    knn = KNN(k=1)
    knn.fit(X_train, y_train)
    assert knn._predict([0, 0]) == 0

def test_find_nearest_points():
    X_train = [[1, 1], [2, 2], [3, 3]]
    y_train = [0, 0, 1]
    knn = KNN(k=2)
    knn.fit(X_train, y_train)
    assert knn.find_nearest_points([0, 0]) == [[1, 1], [2, 2]]

def test_knn_find_nearest_points_fixed():
    X_train = [
        [2, 3],
        [5, 7],
        [6, 5],
        [7, 5],
        [7, 3],
        [8, 2],
        [11, 15],
        [13, 12],
        [14, 13],
        [17, 19]
    ]
    y_train = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    new_point = [6, 6]
    assert knn.find_nearest_points(new_point) == [[6, 5], [5, 7], [7, 5]]

if __name__ == "__main__":
    pytest.main()
