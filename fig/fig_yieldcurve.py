import matplotlib.pyplot as plt

lsc[['r','raw_r','swap','raw_swap']].plot(style=['-k','ok','-r','or'])
# plt.ylabel('Taux')
plt.legend(['Taux zero Nelson-Seigel','Taux zero bruts',
            'Taux swaps Nelson-Seigel','Taux swaps bruts'],
           loc='lower right')
plt.show()
# plt.savefig('fig/yield_curve.pdf')
