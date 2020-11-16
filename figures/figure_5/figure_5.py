from matplotlib import pyplot as plt
import numpy as np
import matplotlib
hs1 = np.arange(1, 37)
hs2 = hs1*3
hs3 = hs1 * 5

# plt.bar(np.zeros(len(hs1)), 0.25, 1, hs1)
fig, ax = plt.subplots()
fig.set_size_inches(6.5, 3)
ax.vlines(hs1, 1, 2, color='black')
ax.vlines(hs2, 0, 1, color='red')
ax.vlines(hs3, 2, 3, color='green')
# plt.scatter([1 for i in range(len(hs2))], hs2, s = 300, marker='_')
ax.set_xscale('log')
ax.set_xlim(0.9, 37)
ax.set_xticks([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36])
ax.set_yticks([])
ax.set_xticklabels([10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 360])
# ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig('figure_5/figure_5.png', dpi=300)
