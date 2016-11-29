from functools import partial

import pandas as pd
import numpy as np
from numpy import sqrt,exp,log
from scipy.stats import norm,ncx2
from scipy.optimize import root,curve_fit
Φ = norm.cdf
χ2 = ncx2.cdf

from zero import get_ns_params,forward_ns,zero_ns

τ = 3/12

# Get NS params for zero curve and zero forward curve
try:
    zero_params
except NameError:
    [zero_params,_] = get_ns_params()
f_m = partial(forward_ns,**zero_params)
R = partial(zero_ns,**zero_params)


def get_data():
    '''Build data from raw data'''
    lsc = pd.DataFrame()
    lsc['T'] = range(1,11)
    lsc['σ'] = [60.04,63.17,54.78,48.62,44.66,41.34,41.25,41.25,35.15,34.28]
    lsc['σ'] = lsc['σ']/100
    lsc['raw_σ'] = lsc['σ']
    # lsc['cap'] = cap(lsc['T'],lsc['σ'])
    lsc = lsc.set_index('T',drop=False)
    return lsc


def P(T):
    '''Price of zero coupon (no discount if t = 0)'''
    T = np.array(T)
    P = np.exp(-T*R(T))
    return P


def F(T1,T2):
    '''Forward rate (simple compounding) F(T1,T2)'''
    τ = T2 - T1
    F = 1/τ*(P(T1)/P(T2) - 1)
    return F


def atm_strike(T):
    # Although ugly, waay faster when it's properly vectorized.
    T = np.array(T).astype(float)
    try:
        tl = np.max(T)
        l = len(T)
    except TypeError:
        tl = T
        l = 1
    Ts = np.arange(2*τ,tl+τ,τ)
    Ts = np.tile(Ts,(l,1)).T
    Ps = np.ma.array(P(Ts),mask=Ts>T)
    s = Ps.sum(axis=0).data
    atm_strike = 1/τ * (P(τ) - P(T))/s
    return atm_strike


def cap_vol(cap_price,T):
    '''Performs inversion of the cap price to find the black volatility associated with it,
    provided a maturity T

    '''
    f = lambda σ: cap(T,σ=σ) - cap_price
    return root(f,x0=0.3).x


def cap(T,σ,K=None):
    '''Black formula for cap pricing provided a constant volatility σ and maturity.
    K defaults to at-the-money strike rate.

    See Brigo & Mercurio2006, Eq. (1.26) p. 18 for details

    '''
    if K is None:
        K = atm_strike(T)
    cap = 0
    for Ti in np.arange(2*τ,T+τ,τ):
        v = σ*sqrt(Ti - τ)
        Fi = F(Ti-τ,Ti)
        d1 = (log(Fi/K) + v**2/2)/v
        d2 = (log(Fi/K) - v**2/2)/v
        Bl = Fi*Φ(d1) - K*Φ(d2)
        cap += P(Ti)*Bl
    return cap*τ
cap = np.vectorize(cap)


def vec_cap(T,σ,K=None):
    T = np.array(T).astype(float)
    σ = np.array(σ).astype(float)
    if K is None:
        K = atm_strike(T)
    try:
        tl = np.max(T)
        l = len(T)
    except TypeError:
        tl = T
        l = 1
    Ts = np.arange(2*τ,tl+τ,τ)
    Ts = np.tile(Ts,(l,1)).T
    T = np.ma.array(Ts,mask=Ts>T)

    v = sqrt(T - τ)*σ
    Fi = F(T-τ,T)
    d1 = (log(Fi/K) + v**2/2)/v
    d2 = (log(Fi/K) - v**2/2)/v
    Bl = Fi*Φ(d1) - K*Φ(d2)
    caps = τ*P(T)*Bl
    caps = np.ma.array(caps,mask=Ts>T)
    caps = caps.sum(axis=0).data
    return caps

# def f_cir(t,*α):
#     # Cf. Brigo & Mercurio2006, Eq. (3.77) p. 102
#     x0,k,θ,σ = α.values()
#     h = sqrt(k**2 + 2*σ**2)
#     f_cir = 2*k*θ*(exp(t*h) - 1)/(2*h + (k+h)*(exp(t*h) - 1)) + \
#         x0 * 4*h**2*exp(t*h)/(2*h + (k+h)*(exp(t*h) - 1))**2
#     return f_cir


# def φ_cir(t,*α):
#     φ = f_m(t) - f_cir(t,α)
#     return φ


def A(t,T,*α):
    # Cf. Brigo & Mercurio2006, Eq. (3.25) p. 66
    x0,k,θ,σ = α
    h = sqrt(k**2 + 2*σ**2)
    A = (2*h*exp((k+h)*(T-t)/2)/(2*h + (k+h)*exp((T-t)*h - 1)))**(2*k*θ/σ**2)
    return A


def B(t,T,*α):
    # Cf. Brigo & Mercurio2006, Eq. (3.25) p. 66
    x0,k,θ,σ = α
    h = sqrt(k**2 + 2*σ**2)
    B = 2*(exp((T-t)*h)-1)/(2*h + (k+h)*(exp((T-t)*h) - 1))
    return B


def ZBC(T,K,*α):
    x0,k,θ,σ = α
    h = sqrt(k**2 + 2*σ**2)
    a = lambda t,T: A(t,T,*α)
    b = lambda t,T: B(t,T,*α)
    r_hat = 1/b(T,T+τ) * (log(a(T,T+τ)/K) -
                          log((P(T)*a(0,T+τ)*exp(-b(0,T+τ)*x0)) /
                              (P(τ)*a(0,T)*exp(-b(0,T)*x0))))
    # Eq (3.26) p. 67
    ρ = 2*h/(σ**2*(exp(h*T) - 1))
    ψ = (k+h)/σ**2

    # p. 103
    q1 = 2*r_hat*(ρ+ψ+b(T,T+τ))
    df = 4*k*θ/σ**2
    nc1 = 2*ρ**2*x0*exp(h*T)/(ρ + ψ + b(T,τ))

    q2 = 2*r_hat*(ρ+ψ)
    nc2 = 2*ρ**2*x0*exp(h*T)/(ρ + ψ)

    ZBC = P(T+τ)*χ2(q1,df,nc1) - K*P(T)*χ2(q2,df,nc2)
    return ZBC


def ZBP(T,K,*α):
    # p. 56
    ZBP = ZBC(T,K,*α) - P(T+τ) + K*P(T)
    return ZBP


def cap_CIR(T,x0,k,θ,σ):
    K = atm_strike(0,T)
    cap = 0
    α = (x0,k,θ,σ)
    for Ti in np.arange(τ,T,τ):
        cap += (1 + K*τ)*ZBP(Ti,1/(1+K*τ),*α)
    return cap
cap_CIR = np.vectorize(cap_CIR)


if __name__ == '__main__':
    lsc = get_data()
    curve_fit(cap_CIR,lsc['T'],lsc['cap'])
