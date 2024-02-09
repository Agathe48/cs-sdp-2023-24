import os
import sys

from python.metrics import (
    PairsExplained,
    ClusterIntersection
)
import numpy as np
from python.data import (
    Dataloader
)
from python.models import (
    HeuristicModel,
    TwoClustersMIP
)

if __name__ == "__main__":  
    print("Starting Python script for evaluation")

    print("MIP Model - dataset_4:")
    ### First part: test of the MIP model
    data_loader = Dataloader("data/dataset_4")  # Path to test dataset

    X, Y = data_loader.load()

    model = TwoClustersMIP(
        n_clusters=2, n_pieces=5
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    model.fit(X, Y)

    # %Pairs Explained
    pairs_explained = PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X, Y))

    # %Cluster Intersection
    cluster_intersection = ClusterIntersection()

    Z = data_loader.get_ground_truth_labels()
    print("% of pairs well grouped together by the model:")
    print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))

    print("Heuristic Model - dataset_10:")
    ### 2nd part: test of the heuristic model
    data_loader = Dataloader("data/dataset_10")  # Path to test dataset
    X, Y = data_loader.load()

    np.random.seed(123)
    indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    np.random.shuffle(indexes)
    train_indexes = indexes[: int(len(indexes) * 0.8)]
    test_indexes = indexes[int(len(indexes) * 0.8) :]

    X_train = X[train_indexes]
    Y_train = Y[train_indexes]
    model = HeuristicModel(n_clusters=3, n_pieces=5, nb_iterations=5)
    model.fit(X_train, Y_train)

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]
    Z_test = data_loader.get_ground_truth_labels()[test_indexes]

    # Validation on test set
    # %Pairs Explained
    pairs_explained = PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X_test, Y_test))

    # %Cluster Intersection
    cluster_intersection = ClusterIntersection()
    print("% of pairs well grouped together by the model:")
    print(
        "Cluster intersection for all samples:",
        cluster_intersection.from_model(model, X_test, Y_test, Z_test),
    )