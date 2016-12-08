import pandas as pd
import numpy as np

from zero import zero_price as P
from cir import r_process

T = 5                           # Maturity
τ = 1/12                        # Rate of payment

N = 25                          # #simulation

r = 5.94/100                    # mortgage rate
rc = 5.5/100                    # Coupon rate

L0 = 7326596                    # Initial Principal
A = 1/(1+r*τ)
C0 = L0/sum(A**i for i in range(1,60+1))  # Initial coupon


def compute_details(p):
    z = pd.DataFrame()
    z['T'] = np.arange(τ,T+τ,τ)
    z['P'] = P(z['T'])                     # Discount curve

    z['p'] = p                  # Probability of prepayment
    z['C'] = (1 - z['p'].shift(1).fillna(0)).cumprod()
    z['C'] *= C0

    L = np.empty(len(z)+1)
    L[0] = L0
    # See veronesi (8.13) p. 298 for details on the update mechanism
    for i in range(1,len(z)+1):
        L[i] = (1 - z.loc[i-1,'p'] + r/12)*L[i-1] - z.loc[i-i,'C']

    z['InitialL'] = L[:-1]
    z['FinalL'] = L[1:]

    z['PP'] = z['p']*z['InitialL']  # Principal Prepaid
    z['MI'] = r/12*z['InitialL']    # Mortgage Interest
    z['SP'] = z['C'] - z['MI']      # Scheduled principal

    z['PTI'] = rc/12*z['InitialL']  # Passthrough interest

    z['CF'] = z['MI'] + z['SP'] + z['PP']  # Cashflows
    z['DCF'] = z['P'] * z['CF']            # Discounted cash flows

    return z


def pv(p):
    '''Present value of the mortgage security given probabiltiy vector of prepayment.'''
    z = compute_details(p)
    pv = z['DCF'].sum()
    return pv


if __name__ == '__main__':
    rs = r_process(T,τ,N)

    cprs = 0.07 + 1.05*np.maximum((0.0594 - (0.00837 + 0.905*rs)),0)
    ps = 1 - (1-cprs)**(1/12)

    p = ps.values[1:,:].T[0]

    # for p in ps.values.T:
    #     answer = pd.DataFrame(index=np.arange(0,T+τ-10e-5,τ))
    #     answer['p'] = p

