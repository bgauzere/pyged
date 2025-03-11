"""
Classes solving a Linear Sum Assigment Problem (LSAP)

Solving a LSAP allows to compute an optimal matching of every pair of elements between two sets $A$ and
$B$, given a cost matrix $C$ where $C_{i, j}$ is the matching cost between $a_i \in A$ and $b_j \in B$

The solution of the LSAP minimizes the costs of $C$
"""

from typing import Protocol, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
# import torch
# import librariesImport
# import gedlibpy
# from sinkdiff.sinkdiff import sinkhorn_d1d2
# from sinkdiff.sink_utils import cost_to_sim


class Solver(Protocol):
    """Solver Protocol
    
    Designs the optimal matching solver classes
    """

    def solve(self, cost_matrix: np.array) -> Tuple[np.array, np.array]:
        """Compute optimal assignment between two sets where matching costs are encoded into `cost_matrix`

        Parameters
        ----------
        cost_matrix : np.array
            The n \times m matrix between the two sets

        Returns
        -------
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
    """LSAP Solver
    
    Solves a Linear Sum Assignment Problem between
    two sets given a matching cost matrix
    """
    def __init__(self):
        pass

    def solve(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the LSAP

        Parameters
        ----------
        C: np.ndarray (2d)
            The matching cost matrix. $C_{i, j}$ is the
            matching cost between elements $i$ and $j$
        
        Returns
        -------
        rho, varrho: np.ndarray
            Optimal matching
            `rho[i]` is the matched index from the second set
            `varrho[i]` is the matched index from the first set
        """
        row_ind, col_ind = linear_sum_assignment(C)
        return row_ind, col_ind


# class SolverLSAPE():
#     def solve(self, C):
#         C_lsape = convert_matrix_to_LSAPE(C)
#         result = gedlibpy.hungarian_LSAPE(C_lsape)
#         # TODO : traiter le retour de result
#         rho = np.array([int(i) for i in result[0]])
#         varrho = np.array([int(i) for i in result[1]])
#         return rho, varrho


# class SolverSinkhorn():
#     def __init__(self, nb_iter=100, eps=1e-2):
#         self.nb_iter = nb_iter
#         self.eps = eps

#     def solve(self, C):
#         C_lsape = convert_matrix_to_LSAPE(C)
#         S = cost_to_sim(torch.from_numpy(C_lsape).float())
#         X, _ = sinkhorn_d1d2(S, self.nb_iter, self.eps)

#         # on inverse pour binariser la matrice
#         results = gedlibpy.hungarian_LSAPE(X.max()-X)
#         rho = np.array([int(i) for i in results[0]])
#         varrho = np.array([int(i) for i in results[1]])
#         return rho, varrho
