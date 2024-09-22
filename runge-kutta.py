import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def kutta_next(xy, tau, f):
    k1 = f(xy)
    k2 = f(xy + tau*k1/2)
    k3 = f(xy + tau*k2/2)
    k4 = f(xy + tau*k3)
    return xy + tau*(k1 + 2*k2 + 2*k3 + k4)/6


def f(xy):
    x, y = xy
    return np.array([alpha * x - beta * x * y,
                     -gamma * y + delta * x * y])


def plot_solution(t, out):
    plt.title('Lotka–Volterra Runge-Kutta approximation')
    plt.xlabel('t')
    plt.plot(t, out[:, 0], label='x (prey)')
    plt.plot(t, out[:, 1], label='y (predator)')
    plt.legend()
    plt.show()


def runge_kutta():
    tau = (tf - ts) / tN
    t = np.linspace(ts, tf, tN)
    xy_all = np.zeros((tN, 2))
    xy_all[0][0], xy_all[0][1] = x0, y0

    pbar = tqdm(range(1, tN), desc='Runge-Kutting')
    for i in pbar:
        xy_all[i] = kutta_next(xy_all[i-1], tau, f)

    plot_solution(t, xy_all)


def load_parameters(file='lotka-volterra-parameters.txt'):
    with open(file) as f:
        res = []
        for i in f:
            res.append(float(i))
    return res


prmtrs = load_parameters()
ts = int(prmtrs[0])
tf = int(prmtrs[1])
tN = int(prmtrs[2])
alpha = prmtrs[3]
beta = prmtrs[4]
gamma = prmtrs[5]
delta = prmtrs[6]
x0 = prmtrs[7]
y0 = prmtrs[8]

runge_kutta()
