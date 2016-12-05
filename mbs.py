from functools import partial

import numpy as np
from numpy import sqrt,exp
import matplotlib.pyplot as plt

from zero import get_ns_params,forward_ns,zero_ns

τ = 1/12

# Get NS params for zero curve and zero forward curve
try:
    zero_params
except NameError:
    [zero_params,_] = get_ns_params()
f_m = partial(forward_ns,**zero_params)
R = partial(zero_ns,**zero_params)


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


def φ_cir(t,*α):
    φ = f_m(t) - f_cir(t,*α)
    return φ


def CIR_process(T,x0,k,θ,σ):
    t = np.arange(0,T+τ,τ)
    x = np.empty_like(t)
    for i in range(len(x)):
        if i == 0:
            x[i] = x0
            continue
        x[i] = x[i-1] + k*(θ-x[i-i])*τ + σ*sqrt(x[i-1])*np.random.normal(0,1)

    r = x + φ_cir(t,x0,k,θ,σ)
    return r


if __name__ == '__main__':
    c = coupon(L=100000,r=0.0756,T=5,τ=1/2)
    print(c)
