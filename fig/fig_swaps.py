swap_nelson = nelson(lsc,'swap')
swap_cubic = cubic(lsc,'swap')

lsc['swap'].plot(style='o')
plt.plot(ts,swap_nelson(ts))
plt.plot(ts,swap_cubic(ts))

plt.legend(['Taux bruts','Nelson-Seigel','Splines cubiques'],loc='lower right')
plt.ylabel('Taux swap')
# plt.savefig('fig/fig_swap_rates.pdf')
plt.show()
