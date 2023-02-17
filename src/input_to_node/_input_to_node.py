import numpy as np
from typing import Literal, Union
from pyrcn.base.blocks import InputToNode
from sklearn.utils.validation import _deprecate_positional_args


class SimpleInputToNode(InputToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 input_activation: Literal['tanh', 'identity', 'logistic',
                                           'relu', 'bounded_relu'] = 'tanh',
                 input_scaling: float = 1., bias_scaling: float = 1.,
                 random_state: Union[int, np.random.RandomState, None] = 42) -> None:
        super().__init__(hidden_layer_size=hidden_layer_size,
                         input_activation=input_activation,
                         input_scaling=input_scaling, input_shift=0.,
                         bias_scaling=bias_scaling, bias_shift=0.,
                         random_state=random_state)

    def fit(self, X: np.ndarray, y: None = None) -> InputToNode:
        """
        Fit the SimpleInputToNode. Fit the input weights and initialize bias.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted InputToNode.
        """
        n_features = X.shape[1]
        rs = np.random.RandomState(self.random_state)
        w_in = 2 * rs.randint(0, 2, (self.hidden_layer_size, n_features)) - 1
        self.predefined_input_weights = w_in.T
        w_bias = 2 * rs.randint(0, 2, (self.hidden_layer_size, )) - 1
        self.predefined_bias_weights = w_bias.reshape(-1, 1)
        return super().fit(X, y)


class SparseClusterInputToNode(InputToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500, n_clusters: int = 200,
                 input_activation: Literal['tanh', 'identity', 'logistic',
                                           'relu', 'bounded_relu'] = 'tanh',
                 input_scaling: float = 1., input_shift: float = 0.,
                 bias_scaling: float = 1., bias_shift: float = 0.,
                 random_state: Union[int, np.random.RandomState, None] = 42,
                 cluster_algorithm: Literal[
                     'kmeans', 'minibatch_kmeans', 'bisecting_kmeans',
                     'birch'] = 'minibatch_kmeans') -> None:
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        super().__init__(hidden_layer_size=hidden_layer_size,
                         input_activation=input_activation,
                         input_scaling=input_scaling, input_shift=input_shift,
                         bias_scaling=bias_scaling, bias_shift=bias_shift,
                         random_state=random_state)

    def fit(self, X: np.ndarray, y: None = None) -> InputToNode:
        """
        Fit the ClusterInputToNode. Fit the input weights and initialize bias.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted InputToNode.
        """
        if self.cluster_algorithm == "kmeans":
            from sklearn.cluster import KMeans
            cluster_algorithm = KMeans(
                n_clusters=self.n_clusters, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        elif self.cluster_algorithm == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            cluster_algorithm = MiniBatchKMeans(
                n_clusters=self.n_clusters, n_init=200,
                reassignment_ratio=0, max_no_improvement=50, init='k-means++',
                random_state=self.random_state).fit(np.vstack(X))
        elif self.cluster_algorithm == "bisecting_kmeans":
            from sklearn.cluster import BisectingKMeans
            cluster_algorithm = BisectingKMeans(
                n_clusters=self.n_clusters, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        else:
            raise TypeError
        w_in = np.divide(
            cluster_algorithm.cluster_centers_,
            np.linalg.norm(cluster_algorithm.cluster_centers_,
                           axis=1)[:, None])
        w_in = np.vstack(
            (w_in, np.zeros(shape=(self.hidden_layer_size - self.n_clusters,
                                   w_in.shape[1]))))
        self.predefined_input_weights = w_in.T
        return super().fit(X, y)


class ClusterInputToNode(InputToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 input_activation: Literal['tanh', 'identity', 'logistic',
                                           'relu', 'bounded_relu'] = 'tanh',
                 input_scaling: float = 1., input_shift: float = 0.,
                 bias_scaling: float = 1., bias_shift: float = 0.,
                 random_state: Union[int, np.random.RandomState, None] = 42,
                 cluster_algorithm: Literal[
                     'kmeans', 'minibatch_kmeans', 'bisecting_kmeans',
                     'birch'] = 'minibatch_kmeans') -> None:
        self.cluster_algorithm = cluster_algorithm
        super().__init__(hidden_layer_size=hidden_layer_size,
                         input_activation=input_activation,
                         input_scaling=input_scaling, input_shift=input_shift,
                         bias_scaling=bias_scaling, bias_shift=bias_shift,
                         random_state=random_state)

    def fit(self, X: np.ndarray, y: None = None) -> InputToNode:
        """
        Fit the ClusterInputToNode. Fit the input weights and initialize bias.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted InputToNode.
        """
        if self.cluster_algorithm == "kmeans":
            from sklearn.cluster import KMeans
            cluster_algorithm = KMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        elif self.cluster_algorithm == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            cluster_algorithm = MiniBatchKMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                reassignment_ratio=0, max_no_improvement=50, init='k-means++',
                random_state=self.random_state).fit(np.vstack(X))
        elif self.cluster_algorithm == "bisecting_kmeans":
            from sklearn.cluster import BisectingKMeans
            cluster_algorithm = BisectingKMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        else:
            raise TypeError
        w_in = np.divide(
            cluster_algorithm.cluster_centers_,
            np.linalg.norm(cluster_algorithm.cluster_centers_,
                           axis=1)[:, None])
        self.predefined_input_weights = w_in.T
        return super().fit(X, y)


class ClusterAttentionInputToNode(InputToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 input_activation: Literal['tanh', 'identity', 'logistic',
                                           'relu', 'bounded_relu'] = 'tanh',
                 input_scaling: float = 1., input_shift: float = 0.,
                 bias_scaling: float = 1., bias_shift: float = 0.,
                 random_state: Union[int, np.random.RandomState, None] = 42,
                 cluster_algorithm: Literal[
                     'kmeans', 'minibatch_kmeans', 'bisecting_kmeans',
                     'birch'] = 'minibatch_kmeans',
                 mode: Literal['unipolar', 'bipolar'] = 'unipolar') -> None:
        self.cluster_algorithm = cluster_algorithm
        self.mode = mode
        super().__init__(hidden_layer_size=hidden_layer_size,
                         input_activation=input_activation,
                         input_scaling=input_scaling, input_shift=input_shift,
                         bias_scaling=bias_scaling, bias_shift=bias_shift,
                         random_state=random_state)

    def fit(self, X: np.ndarray, y: None = None) -> InputToNode:
        """
        Fit the ClusterAttentionInputToNode. Fit the input weights and bias
        weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted InputToNode.
        """
        if self.cluster_algorithm == "kmeans":
            from sklearn.cluster import KMeans
            cluster_algorithm = KMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        elif self.cluster_algorithm == "minibatch_kmeans":
            from sklearn.cluster import MiniBatchKMeans
            cluster_algorithm = MiniBatchKMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                reassignment_ratio=0, max_no_improvement=50, init='k-means++',
                random_state=self.random_state).fit(np.vstack(X))
        elif self.cluster_algorithm == "bisecting_kmeans":
            from sklearn.cluster import BisectingKMeans
            cluster_algorithm = BisectingKMeans(
                n_clusters=self.hidden_layer_size, n_init=200,
                init='k-means++', random_state=self.random_state
            ).fit(np.vstack(X))
        else:
            raise TypeError
        w_in = np.divide(
            cluster_algorithm.cluster_centers_,
            np.linalg.norm(cluster_algorithm.cluster_centers_,
                           axis=1)[:, None])
        self.predefined_input_weights = w_in.T
        labels, counts = np.unique(
            cluster_algorithm.labels_, return_counts=True)
        if len(labels) != self.hidden_layer_size:
            for k in range(self.hidden_layer_size):
                if k not in labels:
                    counts = np.insert(counts, k, 0)
                    labels = np.insert(labels, k, k)
        w_bias = counts / len(cluster_algorithm.labels_)
        w_bias = w_bias / np.max(w_bias)
        if self.mode == "bipolar":
            w_bias = (2 * w_bias) - 1
        self.predefined_bias_weights = w_bias.reshape(-1, 1)
        return super().fit(X, y)
