import numpy as np
import scipy
from typing import Literal, Union
from pyrcn.base.blocks import NodeToNode
from sklearn.utils.validation import _deprecate_positional_args


class DLRBNodeToNode(NodeToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 reservoir_activation: Literal['tanh', 'identity',
                                               'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 forward_weight: float = 1., feedback_weight: float = 1.,
                 leakage: float = 1., bidirectional: bool = False) -> None:
        self.forward_weight = forward_weight
        self.feedback_weight = feedback_weight
        super().__init__(hidden_layer_size=hidden_layer_size,
                         reservoir_activation=reservoir_activation,
                         spectral_radius=1., leakage=leakage,
                         bidirectional=bidirectional)

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the DLRBNodeToNode.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted NodeToNode.
        """
        w_fw = np.eye(self.hidden_layer_size, self.hidden_layer_size, k=1)
        w_fw *= self.forward_weight
        w_fb = np.eye(self.hidden_layer_size, self.hidden_layer_size, k=-1)
        w_fb *= self.feedback_weight
        self.predefined_recurrent_weights = scipy.sparse.csr_matrix(w_fw+w_fb)
        return super().fit(X, y)


class DLRNodeToNode(NodeToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 reservoir_activation: Literal['tanh', 'identity',
                                               'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 forward_weight: float = 1., leakage: float = 1.,
                 bidirectional: bool = False) -> None:
        self.forward_weight = forward_weight
        super().__init__(hidden_layer_size=hidden_layer_size,
                         reservoir_activation=reservoir_activation,
                         spectral_radius=1., leakage=leakage,
                         bidirectional=bidirectional)

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the DLRNodeToNode.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted NodeToNode.
        """
        w_rec = np.eye(self.hidden_layer_size, self.hidden_layer_size, k=1)
        w_rec *= self.forward_weight
        self.predefined_recurrent_weights = scipy.sparse.csr_matrix(w_rec)
        return super().fit(X, y)


class SCRNodeToNode(NodeToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 reservoir_activation: Literal['tanh', 'identity',
                                               'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 forward_weight: float = 1., leakage: float = 1.,
                 bidirectional: bool = False) -> None:
        self.forward_weight = forward_weight
        super().__init__(hidden_layer_size=hidden_layer_size,
                         reservoir_activation=reservoir_activation,
                         spectral_radius=1., leakage=leakage,
                         bidirectional=bidirectional)

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the SCRNodeToNode.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        y : None
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : returns a fitted NodeToNode.
        """
        w_rec = np.eye(self.hidden_layer_size, self.hidden_layer_size, k=1)
        w_rec[-1, 0] = 1
        w_rec *= self.forward_weight
        self.predefined_recurrent_weights = scipy.sparse.csr_matrix(w_rec)
        return super().fit(X, y)
