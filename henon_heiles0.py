import numpy as np
import matplotlib.pyplot as plt
import os

respath = os.sep.join(os.path.dirname(__file__).split(os.sep)[0:-1])+os.sep+'res_matrix_0_40_0.npy'

size_training = 20000
t = 40000
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

# network
def sigma(x): return 1/(1+np.exp(-x))

W_res = np.load(respath)
#W_res = np.matrix([[np.random.random() for i in range(4)] for j in range(size_parameters)])
#np.save(respath, W_res)

A_T = sigma(W_res @ data[0:size_training].T)
W_out_T = np.linalg.solve(A_T @ A_T.T, 
                          A_T @ data[1:size_training+1]).T

# testing
xl = np.empty((len(data) - size_training + 1, 4))
xl[0] = data[size_training].reshape(-1, 1).ravel()
for i in range(1, len(xl)):
    xl[i] = W_out_T @ sigma(W_res @ xl[i-1]).ravel()

# plotting
#plt.plot([x[1] for x in np.array(data[0:size_training+1])], [x[3] for x in np.array(data[0:size_training+1])], color=(0,0,1), label='training data')
plt.plot(data[size_training:-1, 1], data[size_training:-1, 3], color=(1,0,0), label='test data')
plt.plot(xl[:,1], xl[:,3], color=(0,1,0), label='predicted data')
plt.xlabel(r'$y$')
plt.ylabel(r'$p_y$')
plt.title(r'Hénon-Heiles-System with one trajectory')
plt.legend(loc='upper right')
plt.show()
