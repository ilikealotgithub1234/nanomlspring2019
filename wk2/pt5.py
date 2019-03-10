import matplotlib.pyplot as plt
import numpy as np
a = -6
b = 6
c = 100000
X = np.linspace(a,b, num = c)
Y = lambda x: x**4 - 3*x**3 + x**2 - 3*x + 10

plt.plot(X, Y(X))
plt.savefig('plot_pow4.png')
