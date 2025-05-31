import numpy as np
import matplotlib.pyplot as plt
import os

respath = os.sep.join(os.path.dirname(__file__).split(os.sep)[0:-1])+os.sep+'res_matrix_0_40_0.npy'

ax = plt.subplot(projection='3d')

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

# sensitivity
A_TA_inv = np.linalg.inv(A_T @ A_T.T)
def sens(u):
    phi=sigma(W_res @ u)
    x=phi.T @ A_TA_inv @ phi
    return x/(1+x)

ax.plot(data[:,1], data[:,3], color=(0,0,1))

e = 1/2*(data[0][0]**2+data[0][1]**2+data[0][2]**2+data[0][3]**2+data[0][0]**2*data[0][1]-1/3*data[0][1]**3)
sensis=[]
si = 100
sj = 50
ylimit, pylimit = plt.xlim(), plt.ylim()
ys = np.linspace(ylimit[0], ylimit[1], si+1, endpoint=False)[1:]
pys = np.linspace(pylimit[0], pylimit[1], sj+1, endpoint=False)[1:]

sensis = np.array([
    [y, py, float(sens(np.array([0, y, np.sqrt(cpx), py])))]
    for y in ys for py in pys
    if (cpx := 2*e-py**2+y**2*(2/3*y-1)) >= 0
])

surf = ax.plot_trisurf(sensis[:, 0], sensis[:, 1], sensis[:, 2], color=(1,0,0))
plt.show()