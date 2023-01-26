import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt



fig, ax = plt.subplots(1, 1)
s = 0.0000001
loc = 0
mean, var, skew, kurt = lognorm.stats(s, loc=loc, moments='mvsk')
x = np.linspace(lognorm.ppf(0.01, s, loc=loc),
                lognorm.ppf(0.99, s, loc=loc), 100)
ax.plot(x, lognorm.pdf(x, s, loc=loc),
       'r-', lw=5, alpha=0.6, label='lognorm pdf')
rv = lognorm(s, loc=loc)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
r = lognorm.rvs(s, loc=loc, size=1000)
print(r)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc='best', frameon=False)
plt.show()