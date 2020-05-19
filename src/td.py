import pandas as pd
import numpy as np
import math
from scipy.linalg import toeplitz
import scipy.linalg as linalg
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.optimize import Bounds
import sys

class TempDisagg:

    def __init__(self, conversion, fr, n_bc, n_fc, method="chow-lin-maxlog", rho_start=0.5, truncated_rho=0,
                 fixed_rho=0.5):
        self.conversion = conversion
        self.fr = fr
        self.n_bc = n_bc
        self.n_fc = n_fc
        self.rho_start = rho_start
        self.method = method
        self.fixed_rho = fixed_rho
        self.n_l = None
        self.truncated_rho = truncated_rho

        self.rho_min = None

    def generate_conversion_matrix(self):

        conversion_weights = np.repeat(1, self.fr).reshape((1, self.fr))
        diagonal_identity = np.identity(self.n_l)
        conversion_matrix = np.kron(diagonal_identity, conversion_weights.T).T

        if self.n_fc > 0:
            conversion_matrix = np.hstack((conversion_matrix, np.zeros((conversion_matrix.shape[0], self.n_fc))))

        if self.n_bc > 0:
            conversion_matrix = np.hstack((np.zeros((conversion_matrix.shape[0], self.n_bc)), conversion_matrix))

        return conversion_matrix

    def fill_off_diag(self, matrix, val):
        for r, row in enumerate(matrix):
            for c, column in enumerate(row):
                if r - c == 1:
                    matrix[r, c] = val
        return matrix

    def calculate_Q(self, pm, rho):
        return (1 / (1 - rho ** 2)) * (rho ** pm)

    def calculate_QLit(self, X, rho):

        n = X.shape[0]

        H = np.identity(n)
        D = np.identity(n)
        H = self.fill_off_diag(H, -1)
        D = self.fill_off_diag(D, -rho)

        Q_Lit = np.linalg.inv(D.T.dot(H.T).dot(H).dot(D))

        return Q_Lit

    def calculate_dyn_adj(self, X, rho):
        n = len(X)
        diag = np.identity(n)
        diag_rho = self.fill_off_diag(diag, -rho)
        # print(X.shape)
        # print(np.linalg.inv(diag_rho).shape)
        # print(np.hstack((rho, np.zeros(n-1))).shape)
        # print(np.hstack((rho, np.zeros(n-1))).reshape(n, 1).shape)
        # print(np.hstack((X.reshape(n,1), np.hstack((rho, np.zeros(n-1))).reshape(n, 1))).shape)
        return np.linalg.inv(diag_rho).dot(np.hstack((X.reshape(n,1), np.hstack((rho, np.zeros(n-1))).reshape(n, 1))))

    def CalcGLS(self, y, X, vcov, stats=False):

        b = y
        A = X
        W = vcov

        m = A.shape[0]
        n = A.shape[1]

        print("inside CalcGLS")
        print(m)
        print(n)

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

        if len(c_bB1) > 1:
            RHS = c_bB1.reshape(len(c_bB1), 1) - C_bB1.dot(v)
        else:
            RHS = c_bB1 - C_bB1.dot(v)

        print("R")
        print(R)
        print("RHS[0")
        print(RHS[0])
        x = solve_triangular(R, RHS[0])

        z["coefficients"] = x

        z["rss"] = u2.T.dot(u2)

        z["s_2"] = z["rss"] / m
        z["logl"] = -m / 2 - m * np.log(2 * math.pi) / 2 - m * np.log(z["s_2"]) / 2 - np.log(np.linalg.det(vcov)) / 2

        if stats:
            z["s_2_gls"] = z["rss"] / (m - n)

            Lt = C_bB1.dot(P1)
            R_inv = np.linalg.inv(R)

            if Lt.shape[1] > 1:
                R_inv = np.append(R_inv, 0).reshape(1, 2)
            C = R_inv.dot(Lt).dot(Lt.T).dot(R_inv.T)

            # z["se"] = np.sqrt(np.identity(math.floor(z["s_2_gls"] * C)))

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
        self.n_l = len(y_l)

        c_matrix = self.generate_conversion_matrix()

        X_l = c_matrix.dot(X.reshape(len(X), 1))
        pm = np.array(toeplitz(np.arange(n)), dtype=np.float64)

        if self.method == "chow-lin-maxlog":
            def objective_fn(rho):
                Q = self.calculate_Q(pm, rho)
                vcov = (c_matrix.dot(Q)).dot(c_matrix.T)
                print("In objective chin-lin: "+str(X_l.shape))
                return -1 * self.CalcGLS(y_l, X_l, vcov)["logl"]
        elif self.method == "chow-lin-minrss-ecotrim":
            def objective_fn(rho):
                Q = rho**pm
                vcov = (c_matrix.dot(Q)).dot(c_matrix.T)
                return self.CalcGLS(y_l, X_l, vcov)["rss"]
        elif self.method == "chow-lin-minrss-quilis":
            def objective_fn(rho):
                Q = self.calculate_Q(pm, rho)
                vcov = (c_matrix.dot(Q)).dot(c_matrix.T)
                return self.CalcGLS(y_l, X_l, vcov)["rss"]
        elif self.method == "litterman-maxlog":
            def objective_fn(rho):
                Q_Lit = self.calculate_QLit(X, rho)
                vcov = (c_matrix.dot(Q_Lit)).dot(c_matrix.T)
                return -1 * self.CalcGLS(y_l, X_l, vcov)["logl"]
        elif self.method == "litterman-minrss":
            def objective_fn(rho):
                Q_Lit = self.calculate_QLit(X, rho)
                vcov = (c_matrix.dot(Q_Lit)).dot(c_matrix.T)
                return self.CalcGLS(y_l, X_l, vcov)["rss"]
        elif self.method == "dynamic-maxlog":
            def objective_fn(rho):
                X_adj = self.calculate_dyn_adj(X, rho)
                X_l_adj = c_matrix.dot(X_adj)
                print("In objective dyn: "+str(X_l_adj.shape))
                Q = self.calculate_Q(pm, rho)
                vcov = (c_matrix.dot(Q)).dot(c_matrix.T)
                return -1 * self.CalcGLS(y_l, X_l_adj, vcov)["logl"]
        elif self.method == "dynamic-minrss":
            def objective_fn(rho):
                X_adj = self.calculate_dyn_adj(X, rho)
                X_l_adj = c_matrix.dot(X_adj)
                Q = self.calculate_Q(pm, rho)
                vcov = (c_matrix.dot(Q)).dot(c_matrix.T)
                return -1 * self.CalcGLS(y_l, X_l_adj, vcov)["rss"]

        else:
            sys.exit("method invalid")

        if self.method in [ "chow-lin-maxlog", "chow-lin-minrss-ecotrim", "chow-lin-minrss-quilis", "litterman-maxlog",
                            "litterman-minrss", "dynamic-maxlog", "dynamic-minrss"]:

            x0 = np.asarray([0.1])
            bounds = Bounds([-0.999], [0.999])
            min_obj = minimize(objective_fn, x0, bounds=bounds)

            self.rho_min = min_obj["x"][0]

            if self.rho_min < self.truncated_rho:
                self.rho_min = self.truncated_rho
            elif self.method in ["fernandez", "rho"]:
                self.rho_min = 0
            elif self.method in ["chow-lin-fixed", "litterman-fixed", "dynamic-fixed"]:
                self.rho_min = self.fixed_rho

        print("self.rho_min ")
        print(self.rho_min)
        if self.method in ["chow-lin-maxlog", "chow-lin-minrss-ecotrim",
                            "chow-lin-minrss-quilis", "chow-lin-fixed",
                            "dynamic-maxlog", "dynamic-minrss",
                            "dynamic-fixed", "ols"]:
            Q_real = self.calculate_Q(pm, self.rho_min)
        elif self.method in ["fernandez", "litterman-maxlog", "litterman-minrss","litterman-fixed"]:
            Q_real = self.calculate_QLit(X, self.rho_min)

        if self.method in ["dynamic-maxlog", "dynamic-minrss", "dynamic-fixed"] and self.rho_min != 0:
            X = self.calculate_dyn_adj(X, self.rho_min)
            print("inside if:"+str(X.shape))
            X_l = c_matrix.dot(X)

        Q_l_real = c_matrix.dot(Q_real).dot(c_matrix.T)
        z = self.CalcGLS(y_l, X_l, Q_l_real, stats=True)

        # print(z["coefficients"].shape)
        # print(X.reshape(len(X), 2).shape)
        print("z['coefficients']:")
        print(z["coefficients"])

        p = X.reshape(X.shape[0], 1).dot(z["coefficients"])
        D = Q_real.dot(c_matrix.T).dot(z["vcov_inv"])

        u_l = y_l.reshape(len(y_l), 1) - c_matrix.dot(p).reshape(len(c_matrix.dot(p)),1)

        y = p.reshape(len(p),1) + D.dot(u_l)

        return y


y_l_data = pd.read_csv("C:/Users/jstep/Documents/y_l.csv")
X_data = pd.read_csv("C:/Users/jstep/Documents/X.csv")

y_l = np.asarray(y_l_data["V1"])
X = np.asarray(X_data["exports.q"])

# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2)
# td_obj(X, y_l)

# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="litterman-maxlog")
# td_obj(X, y_l)
#
# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="chow-lin-minrss-ecotrim")
# td_obj(X, y_l)
# print(td_obj.rho_min)
#
# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="chow-lin-minrss-quilis")
# print(td_obj(X, y_l))
# print(td_obj.rho_min)
#
# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="litterman-maxlog")
# print(td_obj(X, y_l))
# print(td_obj.rho_min)
#
# td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="litterman-minrss")
# print(td_obj(X, y_l))
# print(td_obj.rho_min)

td_obj = TempDisagg(conversion="sum", fr=4, n_bc=12, n_fc=2, method="dynamic-maxlog")
print(td_obj(X, y_l))
print(td_obj.rho_min)
