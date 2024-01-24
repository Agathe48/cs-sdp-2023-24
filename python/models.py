import pickle
from abc import abstractmethod
import gurobipy as gp
from gurobipy import GRB

import numpy as np

class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)
        

class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, n_pairs, n_criteria):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        n_pairs: int
            Number of pairs in X and in Y.
        n_criteria: int
            Number of critera in X and in Y.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.n_pairs = n_pairs
        self.n_criteria = n_criteria
        self.model = self.instantiate()

    def compute_score(self, x, cluster, evaluate:bool = False):
        get_val = (lambda v: v.X) if evaluate else (lambda v: v)
        score = 0

        width_interval = 1 / (self.n_pieces-1)
        for i in range (self.n_criteria):
            x_i = x[i]
            for l in range(1, self.n_pieces):
                x_i_l = l*width_interval
                if x_i <= x_i_l:
                    break

            a = (get_val(self.score_k_i_l[cluster][i][l]) - get_val(self.score_k_i_l[cluster][i][l-1]))/width_interval
            offset = x_i_l - width_interval
            b = get_val(self.score_k_i_l[cluster][i][l-1])
            
            # Equation de la droite affine => impact du critère i sur le score
            score += a*(x_i-offset) + b
        return score

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        # Gurobi model instantiation
        model = gp.Model("TwoClustersMIP")

        return model

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        ### Variables ###

        # Surestimation error on x
        self.sigma_x_plus = [self.model.addVar(name=f"sigma_x_+_{j}") for j in range(self.n_pairs)]
        # Underestimation error on x
        self.sigma_x_minus = [self.model.addVar(name=f"sigma_x_-_{j}") for j in range(self.n_pairs)]
        # Surestimation error on y
        self.sigma_y_plus = [self.model.addVar(name=f"sigma_y_+_{j}") for j in range(self.n_pairs)]
        # Underestimation error on y
        self.sigma_y_minus = [self.model.addVar(name=f"sigma_y_-_{j}") for j in range(self.n_pairs)]
        M = self.n_criteria

        self.delta_j_k = []
        for j in range(self.n_pairs):
            list_temp = []
            for k in range (self.n_clusters):
                list_temp.append(self.model.addVar(vtype=gp.GRB.BINARY, name=f"delta_j_k_{j}_{k}"))
            self.delta_j_k.append(list_temp)

        self.score_k_i_l = []
        for k in range(self.n_clusters):
            list_temp_i = []
            for i in range (self.n_criteria):
                list_temp_l = []
                for l in range (self.n_pieces):
                    list_temp_l.append(self.model.addVar(lb=0, ub=1, vtype='C', name=f"score_k_i_l_{k}_{i}_{l}"))
                list_temp_i.append(list_temp_l)
            self.score_k_i_l.append(list_temp_i)
    
        # Update of the model
        self.model.update()

        ### Objective function ###
        self.model.setObjective(sum(self.sigma_x_plus) + sum(self.sigma_x_minus) + sum(self.sigma_y_plus) + sum(self.sigma_y_minus), GRB.MINIMIZE) 

        # Contrainte 1 : Origine à 0
        for k in range(self.n_clusters):
            for i in range(self.n_criteria):
                self.model.addConstr(self.score_k_i_l[k][i][0]==0)

        # Contrainte 2 : Normalisation
        for k in range (self.n_clusters):
            for i in range (self.n_criteria):
                self.model.addConstr(sum(self.score_k_i_l[k][i])==1)
        
        # Contrainte 3 : Croissance des fonctions par morceaux
        for k in range(self.n_clusters):
            for i in range(self.n_criteria):
                for l in range(self.n_pieces-1):
                    self.model.addConstr(self.score_k_i_l[k][i][l+1] >= self.score_k_i_l[k][i][l])

        # Contrainte 4 : Contrainte sur les erreurs avec les points de cassure
        for j in range(self.n_pairs):
            x = X[j]
            y = Y[j]
            for k in range (self.n_clusters):
                score_x = self.compute_score(x, cluster=k)
                score_y = self.compute_score(y, cluster=k)
                self.model.addConstr((1-self.delta_j_k[j][k])*M + (score_x - self.sigma_x_plus[j] + self.sigma_x_minus[j]) - (score_y - self.sigma_y_plus[j] + self.sigma_y_minus[j]) >= 0)
        
        # Contrainte 5 : Appartenance à au moins l'un des deux clusters
        for j in range (self.n_pairs):
            self.model.addConstr(sum(self.delta_j_k[j])>=1)

        # Update of the model
        self.model.update()

        # Solve it!
        self.model.optimize()

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        results = []
        for x in X:
            list_temp_k = []
            for k in range(self.n_clusters):
                list_temp_k.append(self.compute_score(x, cluster=k, evaluate=True))
            results.append(list_temp_k)
        return np.array(results)


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
