import numpy as np
import matplotlib.pyplot as plt

t = 100000
h=0.001

# generate data of one trajectory using Explicit Euler
def generate_data(x0, h=0.001, t=20000):
    data = [np.array(x0, dtype=np.float64)]
    for i in range(t):
        data.append(np.array([data[i][0]+h*data[i][2],
                              data[i][1]+h*data[i][3],
                              data[i][2]+h*(-data[i][0]-2*data[i][0]*data[i][1]),
                              data[i][3]+h*(-data[i][1]-data[i][0]**2+data[i][1]**2)], 
                              dtype=np.float64))
    return np.array(data)

data = generate_data(x0=[0,1/2,1/np.sqrt(12), 1/np.sqrt(12)], h=h, t=t)

plt.plot(data[:, 0], data[:, 1], color=(1,0,0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Hénon-Heiles-System with Explicit Euler')
plt.show()