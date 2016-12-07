from functools import partial

import numpy as np
from numpy import sqrt,exp

from zero import forward_zero_curve as f_m
from black import cir_params, load_cir_params


def coupon(L,r,T,τ=1/12):
    i = np.arange(1,1/τ*T + 1)
    coupon = L/sum(1/((1+τ*r)**i))
    return coupon


def f_cir(t,x0,k,θ,σ):
    # Cf. Brigo & Mercurio2006, Eq. (3.77) p. 102
    # x0,k,θ,σ = α
    h = sqrt(k**2 + 2*σ**2)
    f_cir = 2*k*θ*(exp(t*h) - 1)/(2*h + (k+h)*(exp(t*h) - 1)) + \
        x0 * 4*h**2*exp(t*h)/(2*h + (k+h)*(exp(t*h) - 1))**2
    return f_cir


def φ_cir(t,**α):
    φ = f_m(t) - f_cir(t,**α)
    return φ


def x_process(T,τ,x0,k,θ,σ):
    # Glasserman p.124
    d = 4*θ*k/σ**2
    x = np.empty(1/τ*T)
    c = σ**2*(1-exp(-k*τ))/(4*k)
    x[0] = x0
    for i in np.arange(1,1/τ*T):
        λ = x[i-1]*exp(-k*τ)/c
        x[i] = c*np.random.noncentral_chisquare(d-1,λ)
    return x


def CIR_process(T,**α):
    t = np.arange(1/τ*T)
    r = x_process(T,**α) + φ_cir(t,**α)
    return r


if __name__ == '__main__':
    if cir_params is None:
        load_cir_params()
