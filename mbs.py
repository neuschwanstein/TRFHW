from scipy import stats
import pandas as pd
import numpy as np

from zero import zero_price as P, marginal_zero_change
from cir import r_process

τ = 1/12
T = 5                          # Maturity
N = 5                       # #simulation
r = 5.93/100                    # mortgage rate
rc = 5.5/100                    # Coupon rate

L0 = 7326596                    # Initial Principal
A = 1/(1+r*τ)
C0 = L0/sum(A**i for i in range(1,60+1))  # Initial coupon


def compute_details(p):
    z = pd.DataFrame()
    z['T'] = np.arange(τ,T+τ,τ)
    z['P'] = P(z['T'])                     # Discount curve

    p = p[:-1]
    z['p'] = p                  # Probability of prepayment
    z['C'] = (1-p).cumprod()
    z['C'] *= C0

    L = np.empty(len(z))
    L[0] = L0
    # See veronesi (8.13) p. 298 for details on the update mechanism
    for i in range(1,len(z)):
        L[i] = (1 - p[i-1] + r/12)*L[i-1] - z.loc[i-1,'C']

    z['L'] = L

    z['PP'] = z['p']*z['L']  # Principal Prepaid
    z['MI'] = r/12*z['L']    # Mortgage Interest
    z['SP'] = z['C'] - z['MI']      # Scheduled principal

    z['PTI'] = rc/12*z['L']  # Passthrough interest

    z['CF'] = z['PTI'] + z['SP'] + z['PP']  # Cashflows
    z['DCF'] = z['P'] * z['CF']            # Discounted cash flows

    z['PO'] = z['SP']+z['PP']
    z['IO'] = z['PTI']

    return z


def table_details(p):
    z = compute_details(p)
    l_ch = {'T':'$T$','p':'$p$','C':'Coupon','L':'Principal',
            'PP':'Princ. prépayé','MI':'Intérêt',
            'SP':'Princ. anticipé','PTI':'Int. transféré','P':'$P(0,t)$'}
    for old,new in l_ch.items():
        z[new] = z[old]

    to_money = lambda s: r"\num{%0.2f}\$" % s
    to_num4 = lambda s: r"\num{%0.4f}" % s
    to_num2 = lambda s: r"\num{%0.2f}" % s

    formatters = {l_ch['C']:to_money,l_ch['p']:to_num4,l_ch['L']:to_money,
                  l_ch['PP']:to_money,l_ch['MI']:to_money,l_ch['PTI']:to_money,
                  l_ch['SP']:to_money,l_ch['P']:to_num4,l_ch['T']:to_num2}

    table = z[:36].to_latex(formatters=formatters,
                            columns=[l_ch[v] for v in ['T','p','L','C','PP','PTI','MI','SP','P']],
                            column_format=len(formatters)*'c',
                            index=False,
                            escape=False)
    with open('fig/table.tex','w') as f:
        f.write(table)


def faster(ps,no_sum=False):
    Ts = np.arange(τ,T+τ,τ)
    Ps = P(Ts)
    try:
        ps = ps.values
    except AttributeError:
        pass                    # Already a numpy array
    ps = ps[:-1]
    cs = C0*(1-ps).cumprod(axis=0)

    L = np.empty_like(cs)
    L[0,:] = L0
    for i in range(1,len(Ts)):
        L[i,:] = (1 - ps[i-1,:] + r/12)*L[i-1,:] - cs[i-1,:]

    principal_prepaid = ps*L
    mortgage_interest = r/12*L
    scheduled_principal = cs - mortgage_interest

    passthrough_interest = rc/12*L
    # cashflows = passthrough_interest + scheduled_principal + principal_prepaid
    # discounted_cf = Ps*cashflows

    io = (Ps*passthrough_interest.T).T
    po = (Ps*(scheduled_principal + principal_prepaid).T).T

    if no_sum:
        return io,po
    else:
        return io.sum(axis=0),po.sum(axis=0)

    # io = (Ps*passthrough_interest.T).sum(axis=1)
    # po = (Ps*(scheduled_principal + principal_prepaid).T).sum(axis=1)
    # return io,po


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


def get_ps(rs):
    cprs = 0.07 + 1.05*np.maximum((0.0594 - (0.00837 + 0.905*rs)),0)
    ps = 1 - (1-cprs)**(1/12)
    return ps


def duration(delta):
    # Baseline, no shift in the curve
    marginal_zero_change(0)     # Reset baseline
    rs = r_process(T,τ,N)
    ps = get_ps(rs)
    ios,pos = faster(ps)
    io = ios.mean()
    po = pos.mean()

    # Up shift of delta in the zero curve level
    marginal_zero_change(delta)
    rs = r_process(T,τ,N)
    ps = get_ps(rs)
    ios_plus,pos_plus = faster(ps)
    io_plus = ios_plus.mean()
    po_plus = pos_plus.mean()

    # Down shift of delta in the zero curve level
    marginal_zero_change(-delta)
    rs = r_process(T,τ,N)
    ps = get_ps(rs)
    ios_minus,pos_minus = faster(ps)
    io_minus = ios_minus.mean()
    po_minus = pos_minus.mean()

    deriv_io = (io_plus - io_minus)/(2*delta)
    deriv_po = (po_plus - po_minus)/(2*delta)

    dur_io = - 1/io*deriv_io
    dur_po = - 1/po*deriv_po

    return dur_io,dur_po


def _table_iopo(N=50):
    rs = r_process(T,τ,N)
    ps = get_ps(rs)
    ios,pos = faster(ps)

    z = pd.DataFrame(index=['IO','PO'])
    z['Moyenne'] = [ios.mean(),pos.mean()]
    z['Écart-Type'] = [ios.std(),pos.std()]
    z['Asymétrie'] = [stats.skew(ios),stats.skew(pos)]

    to_money = lambda s: r"\num{%s}\$" % str(int(s))
    to_num2 = lambda s: r"\num{%0.2f}" % s

    table = z.to_latex(formatters={'Moyenne':to_money,'Écart-Type':to_num2,'Asymétrie':to_num2},
                       escape=False)
    with open('fig/iopo_table.tex','w') as f:
        f.write(table)


def duration2(N):
    rs = r_process(T,τ,N)
    ios,pos = faster(get_ps(rs))  # Initial values
    io = ios.mean()
    po = pos.mean()

    mbs = (ios+pos).mean()

    rs = rs.values
    mean = rs[1].mean()       # Mean after 1 month
    std = rs[1].std()         # Std after 1 month
    r_up = mean+std
    r_dn = mean-std
    idx_up = np.where(rs[1,:] >= r_up)  # Index where higher than mu+std
    idx_dn = np.where(rs[1,:] <= r_dn)  # Index where lower than mu-std
    rs_up = rs.T[idx_up].T
    rs_dn = rs.T[idx_dn].T
    ps_up = get_ps(rs_up)
    ps_dn = get_ps(rs_dn)

    io_up,po_up = faster(ps_up,no_sum=True)
    io_dn,po_dn = faster(ps_dn,no_sum=True)

    mbs_up = (io_up[1:]+po_up[1:]).sum(axis=0).mean()
    mbs_dn = (io_dn[1:]+po_dn[1:]).sum(axis=0).mean()
    io_up = io_up[1:].sum(axis=0).mean()
    io_dn = io_dn[1:].sum(axis=0).mean()
    po_up = po_up[1:].sum(axis=0).mean()
    po_dn = po_dn[1:].sum(axis=0).mean()

    io_dur = -1/io * (io_up - io_dn)/(rs_up[1].mean() - rs_dn[1].mean())
    po_dur = -1/po * (po_up - po_dn)/(rs_up[1].mean() - rs_dn[1].mean())
    mbs_dur = -1/mbs * (mbs_up - mbs_dn)/(rs_up[1].mean() - rs_dn[1].mean())

    print(rs_up[1].mean())
    print(rs_dn[1].mean())
    return io_dur,po_dur,mbs_dur


if __name__ == '__main__':
    rs = r_process(T,τ,N)

    cprs = 0.07 + 1.05*np.maximum((0.0594 - (0.00837 + 0.905*rs)),0)
    ps = 1 - (1-cprs)**(1/12)

    ios,pos = faster(ps)
    print(ios.mean())
    print(pos.mean())
