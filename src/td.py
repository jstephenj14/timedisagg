import pandas as pd
import numpy as np
import math
from scipy.linalg import toeplitz
import scipy.linalg as linalg
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.optimize import Bounds


class TempDisagg:

    def __init__(self, conversion, fr, n_bc, n_fc, rho = 0.5):
        self.conversion = conversion
        self.fr = fr
        self.n_bc = n_bc
        self.n_fc = n_fc
        self.rho = rho
        self.n_l = None

    def generate_conversion_matrix(self):

        conversion_weights = np.repeat(1, self.fr).reshape((1, self.fr))
        diagonal_identity = np.identity(self.n_l)
        conversion_matrix = np.kron(diagonal_identity, conversion_weights.T).T

        if self.n_fc > 0:
            conversion_matrix = np.hstack((conversion_matrix, np.zeros((conversion_matrix.shape[0], self.n_fc))))

        if self.n_bc > 0:
            conversion_matrix = np.hstack((np.zeros((conversion_matrix.shape[0], self.n_bc)), conversion_matrix))

        return conversion_matrix

    def calculate_Q(self, pm):
        return (1 / (1 - self.rho ** 2)) * (self.rho ** pm)

    def calculate_QLit(self, X):

        """TODO"""

        n = X.shape[0]

        # calclation of S
        H = np.identity(n)
        D = np.identity(n)
        # diag(D[2:nrow(D), 1:(ncol(D) - 1)]) < - -1
        # diag(H[2:nrow(H), 1:(ncol(H) - 1)]) < - -rho
        # Q_Lit < - solve(t(D) % * % t(H) % * % H % * % D)


        # output
        Q_Lit

        return (1 / (1 - self.rho ** 2)) * (self.rho ** pm)

    def CalcGLS(self, y, X, vcov, stats=False):

        b = y
        A = X
        W = vcov

        m = A.shape[0]
        n = A.shape[1]

        B = np.linalg.cholesky(W)

        Q, R = linalg.qr(X)
        R = R[~np.all(R == 0, axis=1)]

        c_bB = Q.T.dot(b)
        c_bB1 = c_bB[0:n]
        c_bB2 = c_bB[n:m]

        C_bB = Q.T.dot(B)
        C_bB1 = C_bB[0:n]
        C_bB2 = C_bB[n:m]

        C_bB2_T = C_bB2.T
        ft_C_bB2 = np.flip(C_bB2_T[0:C_bB2_T.shape[0], 0:C_bB2_T.shape[1]])

        PP, SS = linalg.qr(ft_C_bB2)
        SS = SS[~np.all(SS == 0, axis=1)]

        P = np.flip(PP[0:PP.shape[0], 0:PP.shape[1]])
        S = np.flip(SS[0:SS.shape[0], 0:SS.shape[1]]).T

        P1 = P[:, 0:n]
        P2 = P[:, n:m]

        u2 = solve_triangular(S, c_bB2)
        v = P2.dot(u2)
        v = v.reshape(len(v), 1)

        z = dict()

        x = solve_triangular(R, c_bB1 - C_bB1.dot(v))

        z["coefficients"] = x

        z["rss"] = u2.T.dot(u2)

        z["s_2"] = z["rss"] / m
        z["logl"] = -m / 2 - m * np.log(2 * math.pi) / 2 - m * np.log(z["s_2"]) / 2 - np.log(np.linalg.det(vcov)) / 2

        if stats:
            z["s_2_gls"] = z["rss"] / (m - n)

            Lt = C_bB1.dot(P1)
            R_inv = solve_triangular(R, np.identity(n))
            C = R_inv.dot(Lt).dot(Lt.T).dot(R_inv.T)

            z["se"] = np.sqrt(np.identity(math.floor(z["s_2_gls"] * C)))

            vcov_inv = np.linalg.inv(vcov)
            e = np.repeat(1, m).reshape((m, 1))

            y_bar = ((e.T).dot(vcov_inv).dot(y)) / ((e.T).dot(vcov_inv).dot(e))

            z["tss"] = ((y.reshape(len(y), 1) - y_bar).T).dot(vcov_inv).dot(y.reshape(len(y), 1) - y_bar)

            z["rank"] = n
            z["df"] = m - n
            z["r_squared"] = 1 - z["rss"] / z["tss"]
            z["adj_r_squared"] = 1 - (z["rss"] * (m - 1)) / (z["tss"] * (m - n))
            z["aic"] = np.log(z["rss"] / m) + 2 * (n / m)
            z["bic"] = np.log(z["rss"] / m) + ((n / m) * np.log(m))
            z["vcov_inv"] = vcov_inv

        return z

    def __call__(self, X, y_l):

        n = len(X)
        n_l = len(y_l)

        c_matrix = self.generate_conversion_matrix()

        X_l = c_matrix.dot(X.reshape(len(X), 1))
        pm = np.array(toeplitz(np.arange(n)), dtype=np.float64)

        # Q = self.calculate_Q(pm)
        # vcov = (c_matrix.dot(Q)).dot(c_matrix.T)



