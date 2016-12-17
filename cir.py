import pandas as pd
import numpy as np
from numpy import sqrt,exp,log
from scipy.stats import norm,ncx2
from scipy.optimize import root,curve_fit

from zero import zero_price as P, forward_rate as F, forward_zero_curve as f_m, zero_curve as R

import matplotlib.pyplot as plt

τ = 3/12
Φ = norm.cdf
χ2 = ncx2.cdf

if 'cir_params' not in locals():
    cir_params = None


def get_data():
    '''Build data from raw data'''
    lsc = pd.DataFrame()
    lsc['T'] = range(1,11)
    lsc['σ'] = [60.04,63.17,54.78,48.62,44.66,41.34,41.25,41.25,35.15,34.28]
    lsc['σ'] = lsc['σ']/100
    lsc['raw_σ'] = lsc['σ']
    lsc['cap'] = cap(lsc['T'],lsc['σ'])
    lsc['raw_cap'] = lsc['cap']
    lsc = lsc.set_index('T',drop=False)
    return lsc


def f_cir(t,x0,k,θ,σ):
    # Cf. Brigo & Mercurio2006, Eq. (3.77) p. 102
    h = sqrt(k**2 + 2*σ**2)
    num1 = 2*k*θ*(exp(t*h) - 1)
    den1 = 2*h + (k+h)*(exp(t*h) - 1)
    num2 = 4*h**2*exp(t*h)
    den2 = (2*h + (k+h)*(exp(t*h) - 1))**2
    # f_cir = 2*k*θ*(exp(t*h) - 1)/(2*h + (k+h)*(exp(t*h) - 1)) + \
    # x0 * 4*h**2*exp(t*h)/(2*h + (k+h)*(exp(t*h) - 1))**2
    f_cir = num1/den1 + x0*num2/den2
    return f_cir


def swap_rate(T):
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
cap_vol = np.vectorize(cap_vol)


def cap(T,σ,K=None):
    '''Black formula for cap pricing provided a constant volatility σ and maturity.
    K defaults to at-the-money strike rate.

    See Brigo & Mercurio2006, Eq. (1.26) p. 18 for details

    '''
    T = np.array(T).astype(float)
    σ = np.array(σ).astype(float)
    if K is None:
        K = swap_rate(T)
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
    caplets = τ*P(T)*Bl
    caplets = np.ma.array(caplets,mask=Ts>T)
    cap = caplets.sum(axis=0).data
    return cap


def A(t,T,*α):
    # Cf. Brigo & Mercurio2006, Eq. (3.25) p. 66
    x0,k,θ,σ = α
    h = sqrt(k**2 + 2*σ**2)
    # A = (2*h*exp((k+h)*(T-t)/2)/(2*h + (k+h)*exp((T-t)*h - 1)))**(2*k*θ/σ**2)
    num = 2*h*exp((k+h)*(T-t)/2)
    den = 2*h + (k+h)*(exp((T-t)*h) - 1)
    A = (num/den)**(2*k*θ/σ**2)
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
                              (P(T+τ)*a(0,T)*exp(-b(0,T)*x0))))
    # Eq (3.26) p. 67
    ρ = 2*h/(σ**2*(exp(h*T) - 1))
    ψ = (k+h)/σ**2

    # p. 103
    q1 = 2*r_hat*(ρ+ψ+b(T,T+τ))
    df = 4*k*θ/σ**2
    nc1 = 2*ρ**2*x0*exp(h*T)/(ρ + ψ + b(T,T+τ))

    q2 = 2*r_hat*(ρ+ψ)
    nc2 = 2*ρ**2*x0*exp(h*T)/(ρ + ψ)

    ZBC = P(T+τ)*χ2(q1,df,nc1) - K*P(T)*χ2(q2,df,nc2)
    return ZBC


def ZBP(T,K,*α):
    # Put call parity p. 56
    ZBP = ZBC(T,K,*α) - P(T+τ) + K*P(T)
    return ZBP


def cap_CIR(T,x0,k,θ,σ):
    '''Value of a cap under the CIR++ model. See B&M p. 104, eq. (3.80)

    T: maturity of the cap.'''
    T = np.array(T).astype(float)
    try:
        tl = np.max(T)
        l = len(T)
    except TypeError:
        tl = T
        l = 1
    K = swap_rate(T)
    α = (x0,k,θ,σ)
    # α = {'x0':x0,'k':k,'θ':θ,'σ

    Ts = np.arange(τ,tl,τ)
    Ts = np.tile(Ts,(l,1)).T

    caplets = (1+K*τ)*ZBP(Ts,1/(1+K*τ),*α)
    caplets = np.ma.array(caplets,mask=Ts>=T)
    cap = caplets.sum(axis=0).data
    return cap


def load_cir_params():
    global cir_params
    global x0,k,θ,σ

    lsc = get_data()

    # Initial guess for parameters
    p0 = np.array([0.0001,0.6,0.01,0.08])
    # p0 = None
    # Inverse weight (precision) of measurements
    # sigma = [1,1/12,1/2,1,1,1,1,1/8,1/4,1/4]
    # Fitting of cap prices
    (cir_params,_) = curve_fit(cap_CIR,lsc['T'],lsc['cap'],
                               bounds=([-np.inf,-np.inf,-np.inf,0],
                                       [R(0),np.inf,1.25*f_m(10),0.15]),
                               p0=p0,
                               max_nfev=10000,
                               method='trf')
    #,sigma=sigma,p0=p0,maxfev=10000)

    x0,k,θ,σ = cir_params
    assert(2*k*θ > σ**2)

    args = 'x0 k θ σ'.split()
    cir_params = dict(zip(args,cir_params))

    return cir_params


def _table_cir():
    cir_params = load_cir_params()
    real = ['x0','σ','θ','k']
    idx = ['$x_0$','$\sigma$','$\theta$','$k$']
    z = pd.DataFrame(index=idx)
    z['Paramètres initiaux'] = r"\num{0.0001} \num{0.6} \num{0.01} \num{0.08}".split()
    z['Borne inférieure'] = "$-\infty$ $-\infty$ $-\infty$ 0".split()
    z['Borne supérieure'] = "$R(0)$ $\infty$ $1.25f(0,10)$ 0.15".split()
    # to_num4 = lambda s: r"\num{%0.4f}" % s
    to_num = lambda s: r"\num{%s}" % str(s)
    z['Paramètres obtenus'] = [to_num(cir_params[p]) for p in real]

    table = z.to_latex(escape=False)
    with open('fig/cir_table.tex','w') as f:
        f.write(table)


def cir_process(T,τ,N,x0,k,θ,σ):
    '''CIR process implementation.'''
    m = int(T/τ)+1
    x = np.empty((m,N))

    # Glasserman p.124
    d = 4*θ*k/σ**2
    c = σ**2*(1-exp(-k*τ))/(4*k)
    x[0,:] = x0
    for i in range(1,m):
        λ = x[i-1]*exp(-k*τ)/c
        x[i,:] = c*np.random.noncentral_chisquare(d,λ,N)

    lsc = pd.DataFrame(x,columns=['x_' + str(i) for i in range(N)])
    lsc['T'] = np.arange(0,T+τ-1e-5,τ)  # Not too pretty hack...
    lsc = lsc.set_index('T')
    return lsc


def φ(t,**α):
    φ = f_m(t) - f_cir(t,**α)
    return φ


def r_process(T,τ,N,**α):
    if not α:
        α = load_cir_params()
    x = cir_process(T,τ,N,**α)
    correction = φ(x.index.values,**α)
    r = (x.T + correction).T
    return r


def P_cir(T,τ,N,**α):
    r = r_process(T,τ,N,**α)
    P = exp((-r.cumsum(axis=0)*τ)).mean(axis=1)
    return P


def _cap_prices():
    if cir_params is None:
        load_cir_params()

    lsc = get_data()

    # Update of results datastructure
    for t in np.arange(1,11):
        price = cap_CIR(t,**cir_params)
        σ = cap_vol(price,t)
        lsc.loc[t,'cap'] = price
        lsc.loc[t,'σ'] = σ
        lsc.loc[t,'T'] = t
    lsc = lsc.set_index('T',drop=False)
    lsc = lsc.sort_index()

    # lsc[['cap','raw_cap']].plot(style=['-','o'],legend=None)
    lsc[['σ','raw_σ']].plot(style=['-','o'],legend=False)
    plt.xlabel('$T$')
    plt.ylabel('Volatilite')
    # plt.legend(['Prix CIR++','Prix empiriques'],
    #            loc='lower right')
    plt.legend(['Vol. CIR++','Vol. empiriques'],
               loc='upper right')
    plt.axis(ymax=1,ymin=0.25)
    # plt.axis(xmin=0)
    # plt.show()


def _r_processes(N=25):
    r_process(10,1/12,N,**cir_params).plot(legend=False)
    plt.plot([0,10],[0,0],'--k',linewidth=2)
    plt.xlabel('$T$')
    plt.ylabel('Taux court')
    plt.show()


def _mean_r(N=25):
    ts = np.linspace(0,10,10000)
    plt.plot(ts,f_m(ts))
    rs = r_process(10,1/24,5000,**cir_params)
    μ = rs.mean(axis=1)
    # std = rs.std(axis=1)
    # (μ+std).plot()
    (μ).plot()
    # (μ-std).plot()
    plt.legend(['f(0,t)','Moyenne des trajectoires $r(t)$'],
               loc='lower right')
    plt.xlabel('$T$')
    # plt.show()


def _cap_prices_details():
    lsc = get_data()
    l_ch = {'T':'$T$','σ':'$\sigma$','cap':'Prix cap'}
    for old,new in l_ch.items():
        lsc[new] = lsc[old]

    to_money = lambda s: r"\num{%0.6f}\$" % s
    to_num4 = lambda s: r"\num{%0.4f}" % s

    formatters = {l_ch['σ']:to_num4,l_ch['cap']:to_money}

    table = lsc.to_latex(formatters=formatters,
                         columns=l_ch.values(),
                         escape=False,
                         index=False)

    with open('fig/cap_table.tex','w') as f:
        f.write(table)


if __name__ == '__main__':
    load_cir_params()
