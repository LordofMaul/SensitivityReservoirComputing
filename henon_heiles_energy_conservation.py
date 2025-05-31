import numpy as np
import matplotlib.pyplot as plt
import os

def energy(x):
    return 1/2*(x[0]**2+x[1]**2+x[2]**2+x[3]**2)+x[0]**2*x[1]-x[1]**3/3

respath = os.sep.join(os.path.dirname(__file__).split(os.sep)[0:-1])+os.sep+'res_matrix_0_40_0.npy'

t = 100000
size_parameters = 40
h = 0.001

# generate data of one trajectory using rk4
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

plt.figure(figsize=(15, 5))
plt.plot(list(range(len(data))), [energy(x) for x in data], color = (1,0,0))
plt.title('Energy using RK4 on Hénon-Heiles-System', fontsize=16)
plt.xlabel(f"Number of iterations using $h={h}$", fontsize=12)
plt.ylabel("Energy", fontsize=12)

plt.show()