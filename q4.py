"""
Complex Chooser Option  - part 1 with Fixed CriticalValueChooser value (I) - to focus my attention to the closed form Black-Scholes solution
"""

import math
import numpy as np
from math import *
from scipy import stats
from scipy.stats import norm


# define function f that used in Cumulative Bivariate Normal Distribution
def f(X, y, ap, bp, rho):
    r = ap * (2 * X - ap) + bp * (2 * y - bp) + 2 * rho * (X - ap) * (y - bp)
    f = np.exp(r)
    return f

# Cumulative Bivariate Normal Distribution
def CBND(a, b, rho):
    i = j = 0
    denum = 0
    aarray = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
    barray = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
    pi = 3.14159265358979
    sqval = sqrt(2 * (1 - rho ** 2))
    ap = a / sqval
    bp = b / sqval
    if a <= 0 and b <= 0 and rho <= 0:
        sum = 0
        for i in range(1, 5):
            for j in range(1, 5):
                sum = sum + aarray[i] + aarray[j] * f(barray[i], barray[j], ap, bp, rho)
        sqrval2 = sqrt(1 - rho ** 2)
        Bivn = sum * sqrval2 / pi
    elif a <= 0 and b >= 0 and rho >= 0:
        Bivn = norm.cdf(a) - CBND(a, -b, -rho)
    elif a >= 0 and b <= 0 and rho >= 0:
        Bivn = norm.cdf(b) - CBND(-a, b, -rho)
    elif a >= 0 and b <= 0 and rho <= 0:
        Bivn = norm.cdf(a) + norm.cdf(b) - 1 + CBND(-a, -b, rho)
    elif a * b * rho >= 0:
        denum_str = a ** 2 - 2 * rho * a * b + b ** 2
        denum = np.sqrt(denum_str)
        rho1 = (rho * a - b) * np.sign(a) / denum
        rho2 = (rho * b - a) * np.sign(b) / denum
        delta = (1 - np.sign(a) * np.sign(b)) / 4
        Bivn = CBND(a, 0, rho1) + CBND(b, 0, rho2) - delta
    else:
        return 0
    return Bivn

# Complex Chooser Option
def ComplexChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol):
    I = 51.1158
    # calculating di and d2 (focus on Black Shole model)
    d1 = (np.log(S / I) + ((b + vol ** 2) / 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    y1 = (np.log(S / Xc) + ((b + vol ** 2) / 2) * Tc) / (vol * math.sqrt(Tc))
    y2 = (np.log(S / Xp) + ((b + vol ** 2) / 2) * Tp) / (vol * math.sqrt(Tp))
    rho1 = math.sqrt(T / Tc)
    rho2 = math.sqrt(T / Tp)

    # calculating Complex Chooser option price
    val1 = S * np.exp((b - r) * Tc) * CBND(d1, y1, rho1) - Xc * np.exp(-r * Tc) * CBND(d2, y1 - (vol * np.sqrt(Tc)),
                                                                                       rho1)
    val2 = S * np.exp((b - r) * Tp) * CBND(-d1, -y1, rho2) + Xp * np.exp(-r * Tp) * CBND(-d2, -y2 + (vol * np.sqrt(Tp)),
                                                                                         rho2)
    comp = val1 - val2
    print("complex Chooser option is :", comp)
    # return comp

# Asset price = S , strike price call Xc, strike price put Xp, time to expiration call Tc, time to expiration put Tp, interest rate r, volatility vol
S = 49
Xc = 50
Xp = 46
T = 0.25
Tc = 0.5
Tp = 0.5833
r = 0.10
r1 = 0.4
b = 0.05
vol = 0.12
# using various strike price for call option
xc = range(45, 52, 1)
for i in xc:
    ComplexChooser(S, i, Xp, T, Tc, Tp, r, b, vol)

Tc1 = 1
Tp1 = 1
print("increase time of expiration")
ComplexChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol)
ComplexChooser(S, Xc, Xp, T, Tc1, Tp1, r, b, vol)

print("increase interest rate")
ComplexChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol)
ComplexChooser(S, Xc, Xp, T, Tc, Tp, r1, b, vol)


#for comparing the complex chooser result with lattic method
import numpy as np

def binomial_tree(K,T,S0,r,N,u,d,opttype = "C"):
  dt = T /N
  q = (np.exp(r*dt) - d) / (u - d)
  disc = np.exp(-r * dt)

  S = np.zeros(N+1)
  S[0] = S0 * d **N
  for j in range(1, N+1):
    S[j] = S[j-1] * u/d
  C = np.zeros(N+1)
  for j in range(0, N+1):
    C[j] = max(0, S[j]-K)
  for i in np.arange(N, 0, -1):
    for j in range(0,i):
      C[j] = disc * (q*C[j+1]+ (1-q)* C[j])
  return C[0]
K = 45
T = 0.25
S0 = 49
r = 0.10
N = 30
u = 1.1
d = 0.9
print("Europen Call option value by using Binomial Tree is   ", binomial_tree(K,T,S0,r,N,u,d,opttype = "C"))

"""
Complex Chooser Option - part 2 using CriticalValueChooser 
"""
import math
import numpy as np
from math import *
from scipy import stats
from scipy.stats import norm


# define function f that used in Cumulative Bivariate Normal Distribution
def f(X, y, ap, bp, rho):
    r = ap * (2 * X - ap) + bp * (2 * y - bp) + 2 * rho * (X - ap) * (y - bp)
    f = np.exp(r)
    return f

# define function GDelta that used in CriticalValueChooser
def GDelta(Sv, Xc, Tc1, r, b, v, type):
    if type == "C":
        d1 = (np.log(Sv / Xc) + ((r + v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    elif type == "P":
        d1 = (np.log(Sv / Xc) + ((r - v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    return d1


# define function Gblacksholes that used in CriticalValueChooser
def Gblacksholes(Sv, Xc, Tc1, r, b, v, type="C"):
    #calculate d1, d2
    d1 = (np.log(Sv / Xc) + ((r + v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    d2 = d1 - v * np.sqrt(Tc1)
    #choose it is call or put option
    if type == "C":
        result = Sv * norm.cdf(d1) - Xc * np.exp(-r * Tc1) * norm.cdf(d2)
    elif type == "P":
        result = Xc * np.exp(-r * Tc1) * norm.cdf(-d2) - Sv * norm.cdf(-d1)
    return result

# defining criticalValueChooser
def criticalValueChooser(S, Xc, Xp, T, Tc, Tp, r, b, v):
    Sv = S
    Tc1 = Tc - T
    Tp1 = Tp - T
    d1 = (np.log(Sv / Xc) + ((r + v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    d2 = d1 - v * np.sqrt(Tc1)
    ci = Sv * norm.cdf(d1) - Xc * np.exp(-r * Tc1) * norm.cdf(d2)
    pi = Xc * np.exp(-r * Tc1) * norm.cdf(-d2) - Sv * norm.cdf(-d1)
    dc = (np.log(Sv / Xc) + ((r + v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    dp = (np.log(Sv / Xc) + ((r - v ** 2) / 2) * Tc1) / (v * math.sqrt(Tc1))
    yi = ci - pi
    di = dc - dp
    epsilon = 0.000001
    # while abs(yi)> epsilon:
    Sv = Sv - (yi) / di
    ci = Gblacksholes(Sv, Xc, Tc1, r, b, v, type="C")
    pi = Gblacksholes(Sv, Xp, Tp1, r, b, v, type="P")
    dc = GDelta(Sv, Xc, Tc1, r, b, v, type="C")
    dp = GDelta(Sv, Xc, Tp1, r, b, v, type="P")
    yi = ci - pi
    di = dc - dp
    yi = ci - pi
    di = dc - dp
    print("Critical value chooser is ", Sv)
    return Sv

# define Cumulative Bivariate Normal Distribution
def CBND(a, b, rho):
    i = j = 0
    denum = 0
    aarray = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
    barray = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
    pi = 3.14159265358979
    sqval = sqrt(2 * (1 - rho ** 2))
    ap = a / sqval
    bp = b / sqval
    if a <= 0 and b <= 0 and rho <= 0:
        sum = 0
        for i in range(1, 5):
            for j in range(1, 5):
                sum = sum + aarray[i] + aarray[j] * f(barray[i], barray[j], ap, bp, rho)
        sqrval2 = sqrt(1 - rho ** 2)
        Bivn = sum * sqrval2 / pi
        # print("a")
    elif a <= 0 and b >= 0 and rho >= 0:
        Bivn = norm.cdf(a) - CBND(a, -b, -rho)
        # print("b")
    elif a >= 0 and b <= 0 and rho >= 0:
        Bivn = norm.cdf(b) - CBND(-a, b, -rho)
        # print("c")
    elif a >= 0 and b <= 0 and rho <= 0:
        Bivn = norm.cdf(a) + norm.cdf(b) - 1 + CBND(-a, -b, rho)
        # print("d")
    elif a * b * rho >= 0:
        denum_str = a ** 2 - 2 * rho * a * b + b ** 2
        denum = np.sqrt(denum_str)
        rho1 = (rho * a - b) * np.sign(a) / denum
        rho2 = (rho * b - a) * np.sign(b) / denum
        delta = (1 - np.sign(a) * np.sign(b)) / 4
        # print("c")
        Bivn = CBND(a, 0, rho1) + CBND(b, 0, rho2) - delta
    else:
        return 0
    return Bivn


# Complex Chooser Option
def ComplexChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol):
    I = criticalValueChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol)
    # calculating di and d2 (focus on Black Shole model)
    d1 = (np.log(S / I) + ((b + vol ** 2) / 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    y1 = (np.log(S / Xc) + ((b + vol ** 2) / 2) * Tc) / (vol * math.sqrt(Tc))
    y2 = (np.log(S / Xp) + ((b + vol ** 2) / 2) * Tp) / (vol * math.sqrt(Tp))
    rho1 = math.sqrt(T / Tc)
    rho2 = math.sqrt(T / Tp)

    # calculating Complex Chooser option price
    val1 = S * np.exp((b - r) * Tc) * CBND(d1, y1, rho1) - Xc * np.exp(-1 * r * Tc) * CBND(d2, y1 - (vol * np.sqrt(Tc)),
                                                                                           rho1)
    val2 = S * np.exp((b - r) * Tp) * CBND(-1 * d1, -1 * y1, rho2) + Xp * np.exp(-1 * r * Tp) * CBND(-1 * d2, -y2 + (
    vol * np.sqrt(Tp)), rho2)
    comp = val1 - val2
    return comp


S = 50
Xc = 51
Xp = 48
T = 0.25
Tc = 0.5
Tp = 0.5833
r = 0.10
b = 0.05
vol = 0.35

print("complexchooser is: ", ComplexChooser(S, Xc, Xp, T, Tc, Tp, r, b, vol))