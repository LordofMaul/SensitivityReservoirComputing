import numpy as np
import matplotlib.pyplot as plt
import cmath
import os

e = 1/6
n = 8
m = 5
h = 0.001
size_parameters = 40
load_matrix = True
one_time_labeling = 0

respath = os.sep.join(os.path.dirname(__file__).split(os.sep)[0:-1])+os.sep+'res_matrix_0_40_0.npy'

# generate data of many trajectories using rk4
def hhs(v):
    x, y, px, py = v
    return np.array([px, py, -x - 2*x*y, -y - x**2 + y**2])

def next(v,h):
    k_1=hhs(v)
    k_2=hhs(v+h/2*k_1)
    k_3=hhs(v+h/2*k_2)
    k_4=hhs(v+h*k_3)
    return v+h/6*(k_1+2*k_2+2*k_3+k_4)

def trajectory(h=0.01, x0=[0,1/2,1/np.sqrt(12), 1/np.sqrt(12)]):
    data = [np.array(x0, dtype=np.float64)]
    v = next(x0, h)
    data.append(v)
    c = 1
    while c == True:
        v = next(v, h)
        data.append(v)
        if (v[0]+h*v[2]>0)!=(v[0]>0):
            c = 0
    return np.array(data)

alpha_y = (2*np.sqrt(6)*cmath.sqrt(6*e**2 - e) - 12 *e + 1)**(1/3)

ysolutions = [
    (1/2 *(alpha_y + 1/alpha_y + 1)).real,
    (-1/4*(1 - cmath.sqrt(-3)) *alpha_y - (1 + cmath.sqrt(-3))/(4*alpha_y) + 1/2).real,
    (-1/4*(1 + cmath.sqrt(-3)) *alpha_y - (1 - cmath.sqrt(-3))/(4*alpha_y) + 1/2).real
    ] # computational error --> .real
ysolutions.sort()

data_in, data_out = [],[]

for i in range(1,n+1):
    y = ysolutions[0]+(ysolutions[1]-ysolutions[0])/(n+1)*i
    py_extreme = np.sqrt(2*e+y**2*(2/3*y-1))
    for py in np.linspace(-py_extreme, py_extreme, m+1, endpoint=False)[1:]:
        cpx = 2*e-py**2+y**2*(2/3*y-1)
        if cpx >= 0:
            px = np.sqrt(cpx)
            for k in (-1,1):
                x0 = [0, y, px*k, py]
                tj = trajectory(h, x0)
                if one_time_labeling==0:
                    #plt.plot([x[1] for x in tj], [x[3] for x in tj], color=(0,0,1), label='training data')
                    one_time_labeling+=1
                #else:
                    #plt.plot([x[1] for x in tj], [x[3] for x in tj], color=(0,0,1))
                data_in.append(tj[:-1])
                data_out.append(tj[1:])

data_in = np.concatenate(data_in, axis=0)
data_out = np.concatenate(data_out, axis=0)

# network
def sigma(x): return 1/(1+np.exp(-x))

if load_matrix==True:
    W_res = np.load(respath)
else:
    W_res = np.matrix([[np.random.random() for i in range(4)] for j in range(size_parameters)])
    np.save(respath, W_res)
A_T = sigma(W_res @ data_in.T)
W_out_T = np.linalg.solve(A_T @ A_T.T, 
                          A_T @ data_out).T

# testing
c = 0
while c < 10:
    y = ysolutions[0]+np.random.random()*(ysolutions[1]-ysolutions[0])
    py = 2*(np.random.random()-0.5)*np.sqrt(2*e-y**2+2/3*y**3)
    cpx = 2*e-y**2-py**2+2/3*y**3
    if cpx >= 0:
        px = np.sqrt(cpx)
        x0 = [0, y, px*(-1)**np.random.randint(0,2), py]
        xc = np.array([x0]).T
        tj = trajectory(h, x0)
        test_tj=[] 
        for i in range(tj.shape[0]):
            xc = W_out_T @ sigma(W_res @ xc)
            if not (ysolutions[0]<=xc[1]<=ysolutions[1]) or not (-0.6<=xc[3]<=0.6): break
            test_tj.append(xc)
        test_tj=np.squeeze(np.array(test_tj),axis=-1)
        if one_time_labeling==1:
            plt.plot(tj[:,1], tj[:,3], color=(1,0,0), label='test data')
            plt.plot(test_tj[:,1], test_tj[:,3], color=(0,1,0), label='predicted data')
            one_time_labeling+=1
        else:
            plt.plot(tj[:,1], tj[:,3], color=(1,0,0))
            plt.plot(test_tj[:,1], test_tj[:,3], color=(0,1,0))
        c+=1

plt.xlabel(r'$y$')
plt.ylabel(r'$p_y$')
plt.title(r'Hénon-Heiles-System with many trajectories')
plt.legend(loc='upper right')
plt.show()