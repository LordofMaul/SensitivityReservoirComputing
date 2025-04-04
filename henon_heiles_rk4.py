import numpy as np
import matplotlib.pyplot as plt

t = 100000
h=0.001

# generate data of one trajectory using Runge-Kutta-4
def hhs(v):
    return np.array([v[2],v[3],-v[0]-2*v[0]*v[1],-v[1]-v[0]**2+v[1]**2])

def generate_data(x0, h=0.001, t=20000):
    data = [np.array(x0, dtype=np.float64)]
    for i in range(t):
        v = data[i]
        k_1=hhs(v)
        k_2=hhs(v+h/2*k_1)
        k_3=hhs(v+h/2*k_2)
        k_4=hhs(v+h*k_3)
        data.append(v+h/6*(k_1+2*k_2+2*k_3+k_4))
    return np.array(data)

data = generate_data(x0=[0,1/2,1/np.sqrt(12), 1/np.sqrt(12)], h=h, t=t)

plt.plot(data[:, 0], data[:, 1], color=(1,0,0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Hénon-Heiles-System with Runge-Kutta-4')
plt.show()