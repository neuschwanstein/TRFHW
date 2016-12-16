from math import nan
from functools import partial

import pandas as pd
import numpy as np
from numpy import log,exp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text',usetex=True)

τ = 3/12
ns_params = None
delta = 0


def marginal_zero_change(d):
    global delta
    delta = d


def forward_ns(t,β0,β1,β2,β3,θ1,θ2):
    '''Zero Foward rates with Nelson-Seigel'''
    return β0 + \
        β1*exp(-t/θ1) + \
        β2*t/θ1*exp(-t/θ1) + \
        β3*t/θ2*exp(-t/θ2)


def zero_ns(t,β0,β1,β2,β3,θ1,θ2):
    '''Zero rates with Nelson-Seigel'''

    def zero_ns(t,β0,β1,β2,β3,θ1,θ2):
        return β0 + \
            β1*((1-exp(-t/θ1))/(t/θ1)) + \
            β2*((1-exp(-t/θ1))/(t/θ1) - exp(-t/θ1)) + \
            β3*((1-exp(-t/θ2))/(t/θ2) - exp(-t/θ2))

    t = np.array(t).astype(float)
    R = np.zeros_like(t)
    R[t != 0] = zero_ns(t[t != 0],β0,β1,β2,β3,θ1,θ2)
    R[t == 0] = β0 + β1
    return R


def cubic_interp(lsc,key):
    points = lsc[~lsc[key].isnull()]
    t = points['T']
    r = points[key]
    f = interp1d(t,r,kind='cubic')
    return f


def linear_interp(lsc,key):
    points = lsc[~lsc[key].isnull()]
    t = points['T']
    r = points[key]
    f = interp1d(t,r,kind='linear')
    return f


def update_prices_from_swaps(lsc):
    '''Bootstrap prices from swap rates'''
    s = lsc['P'][τ]

    i = 2
    while True:
        T = i*τ
        try:
            m = τ*lsc['swap'][T]
            P = (1 - m*s)/(1+m)  # Eq. (5.45) Veronesi p. 177
            s += P
            lsc['P'][T] = P
        except KeyError:
            break
        i += 1


def update_prices_from_zerorates(lsc):
    lsc['P'] = np.exp(-lsc['T']*lsc['r'])


def update_zerorates_from_prices(lsc):
    lsc['r'] = -1/lsc['T']*log(lsc['P'])


def update_swaps_from_prices(lsc):
    '''Update swap rates from libor rates'''
    libor = lsc['libor']
    lsc['swap'][τ] = libor[τ]
    s = lsc['P'][τ]
    i = 2

    while True:
        T = i*τ
        try:
            P_T = lsc['P'][T]
            s = s+P_T
            swap = 1/τ * (1 - P_T)/s
            lsc['swap'][T] = swap
        except:
            break
        i = i+1


def interpolate_swaps(lsc,T1,T2,method):
    swap_curve = method(lsc,'swap')
    for T in np.arange(T1,T2+τ,τ):
        lsc.loc[T,'swap'] = swap_curve(T)
        lsc.loc[T,'T'] = T
    lsc.sort_index(inplace=True)

    update_prices_from_swaps(lsc)
    update_zerorates_from_prices(lsc)


def update_zerorates_from_zerocurve(lsc,zero_curve,T1,T2):
    for T in np.arange(T1,T2+τ,τ):
        lsc.loc[T,'r'] = zero_curve(T)
        lsc.loc[T,'T'] = T
        lsc.sort_index(inplace=True)

    update_prices_from_zerorates(lsc)
    update_swaps_from_prices(lsc)


def get_data():
    # Build data from raw data
    lsc = pd.DataFrame()
    Ts = [1/12,3/12,6/12,1,2,3,4,5,7,10]
    lsc['T'] = Ts

    lsc['libor'] = [0.16,0.24,0.33] + 7*[nan]
    lsc['libor'] = lsc['libor']/100
    lsc['r'] = 1/lsc['T']*log(1+lsc['T']*lsc['libor'])

    # Create 'raw' reference table for further comparison
    lsc['raw_r'] = lsc['r']

    lsc['swap'] =3*[nan] + [0.26,0.53,0.97,1.41,1.75,2.04,2.27]
    lsc['swap'] = lsc['swap']/100

    # Use maturity for indexing time
    lsc = lsc.set_index('T',drop=False)

    # Update missing entries
    update_prices_from_zerorates(lsc)
    update_swaps_from_prices(lsc)
    lsc['raw_swap'] = lsc['swap']

    return lsc


def interpolate_curve():
    lsc = get_data()

    interpolate_swaps(lsc,T1=9/12,T2=9/12,method=cubic_interp)
    interpolate_swaps(lsc,T1=1,T2=2,method=cubic_interp)
    interpolate_swaps(lsc,T1=2,T2=3,method=cubic_interp)
    interpolate_swaps(lsc,T1=3,T2=4,method=cubic_interp)
    interpolate_swaps(lsc,T1=4,T2=5,method=cubic_interp)
    interpolate_swaps(lsc,T1=5,T2=7,method=cubic_interp)
    interpolate_swaps(lsc,T1=7,T2=10,method=linear_interp)

    return lsc


def load_ns_params():
    global ns_params

    lsc = interpolate_curve()
    sigma = [1/10,1/6,1/10,1/10,1/15] + [1]*(len(lsc)-6) + [1/20]
    [params,cov] = curve_fit(zero_ns,lsc['T'],lsc['r'],
                             max_nfev=8000,
                             method='trf',
                             sigma=sigma)

    args = 'β0 β1 β2 β3 θ1 θ2'.split()
    ns_params = dict(zip(args,params))


def zero_curve(t):
    if ns_params is None:
        load_ns_params()
    return partial(zero_ns,**ns_params)(t) + delta


def forward_zero_curve(t):
    if ns_params is None:
        load_ns_params()
    return partial(forward_ns,**ns_params)(t) + delta


def zero_price(T):
    '''Price of zero coupon'''
    R = zero_curve
    T = np.array(T)
    P = np.exp(-T*R(T))
    return P


def forward_rate(T1,T2):
    '''Forward rate (simple compounding) F(T1,T2)'''
    P = zero_price
    τ = T2 - T1
    F = 1/τ*(P(T1)/P(T2) - 1)
    return F


def _r_fwd():
    t = np.linspace(0,10,10000)
    plt.plot(t,zero_curve(t),t,forward_zero_curve(t))
    plt.legend(['Taux spot',u'Taux forward'],loc='lower right')
    plt.xlabel('$T$')
    plt.show()


def _r_raw():
    lsc = get_data()
    t = np.linspace(0,10.5,10000)
    plt.plot(t,zero_curve(t))
    plt.plot(t,forward_zero_curve(t))
    lsc['raw_r'].plot(style='o')
    lsc['raw_swap'][1:].plot(style='o')
    plt.xlabel('$T$')
    plt.legend(['Taux spot','Taux forward','Taux LIBOR','Taux swap'],
               loc='lower right')
    # plt.show()


def _sr():
    lsc = get_data()
    t = np.linspace(0,10.5,10000)
    plt.plot(t,zero_curve(t))
    plt.plot(t,szero_curve(t))
    lsc['raw_r'].plot(style='o')
    lsc['raw_swap'][1:].plot(style='o')
    plt.xlabel('$T$')
    plt.show()


if __name__ == '__main__':
    load_ns_params()
