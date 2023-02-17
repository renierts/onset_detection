import numpy as np
import scipy
from itertools import pairwise
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


class TransitionNodeToNode(NodeToNode):
    @_deprecate_positional_args
    def __init__(self, *, hidden_layer_size: int = 500,
                 reservoir_activation: Literal['tanh', 'identity',
                                               'logistic', 'relu',
                                               'bounded_relu'] = 'tanh',
                 spectral_radius: float = 1., leakage: float = 1.,
                 bidirectional: bool = False) -> None:
        super().__init__(hidden_layer_size=hidden_layer_size,
                         reservoir_activation=reservoir_activation,
                         spectral_radius=spectral_radius, leakage=leakage,
                         bidirectional=bidirectional)

    def fit(self, X: np.ndarray, y: None = None) -> NodeToNode:
        """
        Fit the TransitionNodeToNode. Fit the recurrent weights as a transition
        matrix.

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
        state_sequence = np.argmax(X, axis=1)
        print(state_sequence)
        w_rec = self._transition_matrix(state_sequence, self.hidden_layer_size)
        print(w_rec)
        sparsity = 1 - np.count_nonzero(w_rec) / w_rec.size
        if sparsity > .5:
            w_rec = scipy.sparse.csr_matrix(w_rec)
        self.predefined_recurrent_weights = w_rec
        return super().fit(X, y)

    @staticmethod
    def _transition_matrix(state_sequence, hidden_layer_size):
        """
        Compute a 2-gram state transition matrix from a given state sequence.

        Parameters
        ----------
        state_sequence : Union[list, np.ndarray]
            List or array of state transitions.

        hidden_layer_size : int
            The hidden layer size, here, the number of individual states.
        Returns
        -------
        M : np.ndarray, shape=(hidden_layer_size, hidden_layer_size)
            Normalized transition probabilities.
        """
        M = np.zeros(shape=(hidden_layer_size, hidden_layer_size))
        for (state_minus_one, state_minus_zero) in pairwise(state_sequence):
            M[state_minus_one][state_minus_zero] += 1
        # now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        return M
