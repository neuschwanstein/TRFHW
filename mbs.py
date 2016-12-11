import pandas as pd
import numpy as np

from zero import zero_price as P
from cir import r_process

τ = 1/12


def compute_details(p):
    z = pd.DataFrame()
    z['T'] = np.arange(τ,T+τ,τ)
    z['P'] = P(z['T'])                     # Discount curve

    z['p'] = p                  # Probability of prepayment
    # z['C'] = (1 - z['p'].shift(1).fillna(0)).cumprod()
    z['C'] = (1-p).cumprod()
    z['C'] *= C0

    L = np.empty(len(z)+1)
    L[0] = L0
    # See veronesi (8.13) p. 298 for details on the update mechanism
    for i in range(1,len(z)+1):
        L[i] = (1 - z.loc[i-1,'p'] + r/12)*L[i-1] - z.loc[i-1,'C']

    z['InitialL'] = L[:-1]
    z['FinalL'] = L[1:]

    z['PP'] = z['p']*z['InitialL']  # Principal Prepaid
    z['MI'] = r/12*z['InitialL']    # Mortgage Interest
    z['SP'] = z['C'] - z['MI']      # Scheduled principal

    z['PTI'] = rc/12*z['InitialL']  # Passthrough interest

    z['CF'] = z['PTI'] + z['SP'] + z['PP']  # Cashflows
    z['DCF'] = z['P'] * z['CF']            # Discounted cash flows

    return z


def faster(ps):
    # ps = ps[τ:].values
    # ps = np.insert(ps,0,0,axis=0)
    # ps = (1-ps).cumprod(axis=0)
    Ts = np.arange(τ,T+τ,τ)
    Ps = P(Ts)
    ps = ps.values
    ps = ps[:-1]
    cs = C0*(1-ps).cumprod(axis=0)

    L = np.empty_like(cs)
    L[0,:] = L0
    for i in range(1,len(Ts)):
        L[i,:] = (1 - ps[i-1,:] + r/12)*L[i-1,:] - cs[i-1,:]

    # initial_L = L[:-1,:]
    # final_L = L[1:,:]

    principal_prepaid = ps*L
    mortgage_interest = r/12*L
    scheduled_principal = cs - mortgage_interest

    passthrough_interest = rc/12*L
    # cashflows = passthrough_interest + scheduled_principal + principal_prepaid
    # discounted_cf = Ps*cashflows

    io = (Ps*passthrough_interest.T).sum(axis=1)
    po = (Ps*(scheduled_principal + principal_prepaid).T).sum(axis=1)
    return io,po


def pv(p):
    '''Present value of the mortgage security given probabiltiy vector of prepayment.'''
    z = compute_details(p)
    pv = z['DCF'].sum()
    return pv


def io(p):
    '''Computes the present value of the claim on the interests provided by the MBS.'''
    z = compute_details(p)
    io = (z['PTI']*z['P']).sum()
    return io


def po(p):
    '''Computes the present value of the claim on the principal scheduled + principal prepaid
    by the MBS.
    '''
    z = compute_details(p)
    po = ((z['SP']+z['PP'])*z['P']).sum()
    return po


if __name__ == '__main__':
    T = 5                           # Maturity
    # τ = 1/12                        # Rate of payment
    N = 100000                         # #simulation
    r = 5.93/100                    # mortgage rate
    rc = 5.5/100                    # Coupon rate

    L0 = 7326596                    # Initial Principal
    A = 1/(1+r*τ)
    C0 = L0/sum(A**i for i in range(1,60+1))  # Initial coupon

    rs = r_process(T,τ,N)

    cprs = 0.07 + 1.05*np.maximum((0.0594 - (0.00837 + 0.905*rs)),0)
    ps = 1 - (1-cprs)**(1/12)

    ios,pos = faster(ps)
