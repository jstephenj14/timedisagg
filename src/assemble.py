
# Assemble the structure as R code

import pandas as pd
import numpy as np
import math
from scipy.linalg import toeplitz
import scipy.linalg as linalg
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.optimize import Bounds

y_l_data = pd.read_csv("C:/Users/jstep/Documents/y_l.csv")
X_data = pd.read_csv("C:/Users/jstep/Documents/X.csv")

y_l = np.asarray(y_l_data["V1"])
X = np.asarray(X_data["exports.q"])

# Set up CalcC
# inputs
n = len(X)
n_l = len(y_l)
conversion = "sum"
fr = 4 # ratio of high-freq to low-freq for eg, for quartertly to yearly it's 4
n_bc = 12 # number of back-casting periods, identified my mismatch of starting X and y
n_fc = 2 # number of fore-casting periods, identified my mismatch of ending X and y

conversion_weights = np.repeat(1, fr).reshape((1,fr))
diagonal_identity = np.identity(n_l)
conversion_weights_T = conversion_weights.T
C = np.kron(diagonal_identity,conversion_weights_T).T

if n_fc > 0:
    C = np.hstack((C,np.zeros((C.shape[0], n_fc))))

if n_bc > 0:
    C = np.hstack((np.zeros((C.shape[0], n_bc)), C))

pm = np.array(toeplitz(np.arange(n)),dtype=np.float64)


rho = 0.5

X_l = C.dot(X.reshape(len(X),1))

# CalcQ = (1 / (1 - rho**2))*(rho**pm)
# vcov = (C.dot(CalcQ)).dot(C.T)

def CalcGLS(y, X, vcov, stats=False):

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
    ft_C_bB2 = np.flip(C_bB2_T[0:C_bB2_T.shape[0],0:C_bB2_T.shape[1]])

    PP, SS = linalg.qr(ft_C_bB2)
    SS = SS[~np.all(SS == 0, axis=1)]

    P = np.flip(PP[0:PP.shape[0], 0:PP.shape[1]])
    S = np.flip(SS[0:SS.shape[0], 0:SS.shape[1]]).T

    P1 = P[:, 0:n]
    P2 = P[:, n:m]

    u2 = solve_triangular(S, c_bB2)
    v = P2.dot(u2)
    v = v.reshape(len(v),1)

    z = dict()

    x = solve_triangular(R, c_bB1 - C_bB1.dot(v)) # z$coefficients

    z["coefficients"] = x

    z["rss"] = u2.T.dot(u2)
    z["s_2"] = z["rss"] / m
    z["logl"] = -m / 2 - m * np.log(2 * math.pi) / 2 - m * np.log(z["s_2"]) / 2 - np.log(np.linalg.det(vcov)) / 2

    if stats:
        z["s_2_gls"] = z["rss"]/(m-n)

        # vcov
        Lt = C_bB1.dot(P1)
        R_inv = solve_triangular(R, np.identity(n))
        C = R_inv.dot(Lt).dot(Lt.T).dot(R_inv.T)

        # return [z_s_2_gls, C]
        # standard errors of coefficients
        z["se"] = np.sqrt(np.identity(math.floor(z["s_2_gls"]*C)))

        # total sum of squares
        vcov_inv = np.linalg.inv(vcov)
        e = np.repeat(1, m).reshape((m, 1))

        # y_bar < - as.numeric(t(e) % * % vcov_inv % * % y / t(e) % * % vcov_inv % * % e)
        # z$tss < - as.numeric(t(y - y_bar) % * % vcov_inv % * % (y - y_bar))

        #return [e,vcov_inv,y]

        y_bar = ((e.T).dot(vcov_inv).dot(y))/((e.T).dot(vcov_inv).dot(e))

        #return [y,y_bar,vcov_inv]
        z["tss"] = ((y.reshape(len(y),1)-y_bar).T).dot(vcov_inv).dot(y.reshape(len(y), 1)-y_bar)

        # z$rank < - n

        z["rank"] = n
        # z$df < - m - n
        z["df"] = m - n
        # z$r.squared < - 1 - z$rss / z$tss
        z["r_squared"] = 1 - z["rss"]/z["tss"]
        # z$adj.r.squared < - 1 - (z$rss * (m - 1)) / (z$tss * (m - n))
        z["adj_r_squared"] = 1 - (z["rss"]*(m-1))/(z["tss"]*(m-n))
        # z$aic < - log(z$rss / m) + 2 * (n / m)
        z["aic"] = np.log(z["rss"]/m) + 2*(n/m)
        # z$bic < - log(z$rss / m) + log(m) * (n / m)
        z["bic"] = np.log(z["rss"]/m) + ((n/m)*np.log(m))
        # z$vcov_inv < - vcov_inv
        z["vcov_inv"] = vcov_inv

    return z


def func_optimize(rho):
    CalcQ = (1 / (1 - rho ** 2)) * (rho ** pm)
    vcov = C.dot(CalcQ).dot(C.T)
    return -CalcGLS(y_l, X_l, vcov)["logl"]


x0 = np.asarray([0.1])
bounds = Bounds([-0.999], [0.999])
min_obj = minimize(func_optimize, x0, bounds=bounds)

rho_min = min_obj["x"][0]
print(rho_min)

# else if (method % in% c(
# "chow-lin-maxlog", "chow-lin-minrss-ecotrim",
# "chow-lin-minrss-quilis", "chow-lin-fixed",
# "dynamic-maxlog", "dynamic-minrss",
# "dynamic-fixed", "ols"
# )) {
# Q < - CalcQ(rho = rho, pm = pm)

Q_real = (1 / (1 - rho_min**2))*(rho_min**pm)

# aggregating Q
# Q_l < - C % * % Q % * % t(C)
Q_l_real = C.dot(Q_real).dot(C.T)


# final GLS estimation (aggregated)
z = CalcGLS(y_l, X_l, Q_l_real, stats=True)

p = X.reshape(len(X), 1).dot(z["coefficients"])

# distribution matrix
#D < - Q % * % t(C) % * % z$vcov_inv

D = Q_real.dot(C.T).dot(z["vcov_inv"])
# low frequency residuals
# u_l < - y_l - C % * % p

u_l = y_l.reshape(len(y_l), 1) - C.dot(p)

# final series
# y < - p + D % * % u_l
y = p + D.dot(u_l)
