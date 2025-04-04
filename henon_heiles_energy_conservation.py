import numpy as np
import matplotlib.pyplot as plt
import cmath

t = 10000
h=0.001

def energy(x):
    return 1/2*(x[0]**2+x[1]**2+x[2]**2+x[3]**2)+x[0]**2*x[1]-x[1]**3/3

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

e = 1/7
alpha_y = (2*np.sqrt(6)*cmath.sqrt(6*e**2 - e) - 12 *e + 1)**(1/3)

ys = [
    (1/2 *(alpha_y + 1/alpha_y + 1)).real,
    (-1/4*(1 - cmath.sqrt(-3)) *alpha_y - (1 + cmath.sqrt(-3))/(4*alpha_y) + 1/2).real,
    (-1/4*(1 + cmath.sqrt(-3)) *alpha_y - (1 - cmath.sqrt(-3))/(4*alpha_y) + 1/2).real
    ]
ys.sort()

y = ys[0]+np.random.random()*7/8*(ys[1]-ys[0])
py = 0.05
cpx=2*e-py**2+y**2*(2/3*y-1)
px = np.sqrt(cpx)


x0 = [0,y,px,py]

data = generate_data(x0, h=h, t=t)

#plt.plot(data[:,0], data[:,1], color=(1,0,0))

for i in range(len(data)):
    plt.plot(list(range(len(data))), [energy(x) for x in data], color = (1,0,0))

plt.show()
