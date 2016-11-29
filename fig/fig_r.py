swap_nelson = nelson(lsc,'r',extended=True)
# swap_cubic = cubic(lsc,'r')

ts = np.linspace(0,10,5000)
lsc['r'].plot(style='o')
plt.plot(ts,swap_nelson(ts))
# plt.plot(ts,swap_cubic(ts))

plt.legend(['Taux interpolés','Nelson-Seigel étendu','Splines cubiques'],loc='lower right')
plt.ylabel('Taux zero')
plt.savefig('fig/fig_zero_rates.pdf')
# plt.show()
