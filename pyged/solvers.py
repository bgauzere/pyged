import torch
from typing import Protocol
from scipy.optimize import linear_sum_assignment
import numpy as np
import librariesImport
import gedlibpy
from sinkdiff.sinkdiff import sinkhorn_d1d2


class Solver(Protocol):
    def solve(self, cost_matrix: np.array) -> tuple[np.array, np.array]:
        """Compute optimal assignment between two sets where matching costs are encoded into cost_matrix

        Parameters
        ----------
        cost_matrix : np.array
            The n \times m matrix between the two sets

        Returns
        --------
        rho, varrho : np.array
        rho[i] indicates the mapping of i onto second set
        varrho[j] indicates the mapping of j onto first set (inverse of rho)
        """
        ...


def convert_matrix_to_LSAPE(C: np.array) -> np.array:
    """
    convert a n+m \times n+m matrix to a n+1 \times m+1 matrix

    Parameters
    ------------
    C : np.array

    Returns
    -----------
    X:np.array
    """
    n = np.argmax(C[:, 0])-1  # on detecte le premier inf
    m = np.argmax(C[0, :])-1
    insertions = np.diag(C[n:, :m])
    deletions = np.diag(C[:n, m:])

    lsape_cost_matrix = np.block([[C[:n, :m], deletions.reshape(-1, 1)],
                                 [insertions.reshape(1, -1), C[-1, -1]]])
    return lsape_cost_matrix


class SolverLSAP():
    def __init__(self):
        pass

    def solve(self, C):
        row_ind, col_ind = linear_sum_assignment(C)
        return col_ind, row_ind[np.argsort(col_ind)]


class SolverLSAPE():
    def solve(self, C):
        C_lsape = convert_matrix_to_LSAPE(C)
        result = gedlibpy.hungarian_LSAPE(C_lsape)
        # TODO : traiter le retour de result
        rho = np.array([int(i) for i in result[0]])
        varrho = np.array([int(i) for i in result[1]])
        return rho, varrho


class SolverSinkhorn():
    def __init__(self, nb_iter=100, eps=1e-2):
        """

        """
        self.nb_iter = nb_iter
        self.eps = eps

    def solve(self, C):
        # TODO : gerer la conversion cost to sim dans sinkdiff
        C_lsape = convert_matrix_to_LSAPE(C)
        S = -torch.from_numpy(C_lsape).float()
        X, _ = sinkhorn_d1d2(S, self.nb_iter, self.eps)
        results = gedlibpy.hungarian_LSAPE(-X)
        rho = np.array([int(i) for i in results[0]])
        varrho = np.array([int(i) for i in results[1]])
        return rho, varrho
