import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular


class RRQR:
    """
    Strong Rank-Revealing QR Decomposition.

    Produces A*Pi = Q*R where:
    - R11 (leading black) is well-conditioned.
    - R22 (training block) is small (low residual energy)
    - Coupling is small
    """

    def __init__(self, f = 2.0, max_iter = 10):
        """
        Parameters:
        f : float
            Tolerance for maximum entry in coupling matrix T.
            If max(abs(T)) <= f, decompisition is considered rank-revealing.
        max_iter : int
            Maximum number of column swap iterations
        """
        self.f = f
        self.max_iter = max_iter
        self.Q = None
        self.R = None
        self.P = None

    def fit(self, A, k = None):
        """
        Compute strong RRQR factorization.

        Parameters:
        A : array_like, shape (m, n)
            Input matrix
        k : int, optional
            Target rank. If None, use full rank
        """
        A = np.array(A, dtype = float)
        m, n = A.shape
        if k is None:
            k = min(m, n)

        # Step 1: initial QRCP Execution
        # standard QR with column pivoting (weak RRQR)
        Q, R, P = qr(A, pivoting=True, mode='full')
        P = np.array(P)

        iter_count = 0
        while iter_count < self.max_iter:
            # Step 2: Partition R into leading and trailing blocks
            
            # R11: leading kxk block
            # R12: trailing k x (n-k) block
            R11 = R[:k, :k]
            R12 = R[:k, :k]

            # Step 3: Compute the coupling matrix T
            # Solve R11 * T = R12 for T (triangular solve)
            # T tells us how much the trailing columns depend on the leading ones
            if R12.shape[1] == 0:
                # if no trailing columns, leave loop 
                break
            T = solve_triangular(R11, R12)

            # Step 4: Check the max entry of T for tolerance
            # Root cause analysis
            i_star, j_star = np.unravel_index(np.abs(T).argmax(), T.shape)
            max_T = np.abs(T[i_star, j_star])

            if max_T <= self.f:
                # Coupling is small enough => decomposition is rank-revealing
                break
            
            # Step 5: Swap columns based on max entry in T
            # Swap columns: i_star in R11 (leading block), j_star in R12 (trailing block)
            col1 = i_star
            col2 = k + j_star
            R[:, [col1, col2]] = R[:, [col2, col1]]     # swap columns
            P[[col1, col2]] = P[[col2, col1]]           # update permuation

            # Step 6: Restore upper triangular structure with a simple QR on affected columns
            # After the column wrap, R is not upper triangular 
            # Re-factor the affected block with QR to zero out the subdiagonal
            rows = slice(i_star, m)
            cols = slice(col1, col2 + 2)
            Q_local, R_local = qr(R[rows, cols], mode = 'economic')
            R[rows, cols] = R_local     # updated affected submatrix
            # Q update is ignored for simplicity

            iter_count += 1

        # Step 7: Recompute Q for the permutated matrix
        # After all swaps, recompute Q to match final R and permutation
        self.Q, R_full = qr(A[:, P], mode = 'full')
        self.R = R_full
        self.P = P
        return self
         
    def transform(self, X):
        """
        Apply RRQR transformation to a new matrix X
        """
        if self.Q is None:
            raise ValueError("Must fit first.")
        return self.Q.T @ X[:, self.P]
    

############# Testing #############
if __name__ == "__main__":
    np.random.seed(42)
    A = np.random.rand(6, 5)
    rrqr = RRQR(f=1.5, max_iter=5)
    rrqr.fit(A, k=3)
    print("Permutation indices:", rrqr.P)
    print("R matrix:\n", rrqr.R)

    