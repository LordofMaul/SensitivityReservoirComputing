import numpy as np
import matplotlib.pyplot as plt

t = 100000
h=0.001

# generate data of one trajectory using Runge-Kutta-4
def hhs(v):
    x, y, px, py = v
    return np.array([px, py, -x - 2*x*y, -y - x**2 + y**2])

def generate_data(x0, h=0.001, t=20000):
    data = np.empty((t + 1, len(x0)))
    data[0] = x0
    for i in range(t):
        v = data[i]
        k1 = hhs(v)
        k2 = hhs(v + h/2 * k1)
        k3 = hhs(v + h/2 * k2)
        k4 = hhs(v + h * k3)
        data[i + 1] = v + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return data

data = generate_data(x0=[0,1/2,1/np.sqrt(12), 1/np.sqrt(12)], h=h, t=t)

plt.plot(data[:, 0], data[:, 1], color=(1,0,0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Hénon-Heiles-System with Runge-Kutta-4')
plt.show()