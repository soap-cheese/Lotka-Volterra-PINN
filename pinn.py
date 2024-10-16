import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from plottin import *


# class Sin(nn.Module):
#     def forward(self, x):
#         return torch.sin(x)


class Net(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),   # 1
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),   # 2
            nn.Tanh(),
            nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.layers_stack(x)


def pde(outt, tt, alpha, beta, gamma, delta):
    x = outt[:, 0].unsqueeze(1)
    y = outt[:, 1].unsqueeze(1)
    dxdt = torch.autograd.grad(x, tt, torch.ones_like(x), create_graph=True,
                               retain_graph=True)[0]
    dydt = torch.autograd.grad(y, tt, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0]

    pde1 = dxdt - alpha * x + beta * x * y
    pde2 = dydt + gamma * y - delta * x * y

    return torch.cat((pde1, pde2), 1)
    # return pde1, pde2


def pdeloss(tt, epoch, x0, y0, param, lmbd, update_array=True):
    outt = the_net(tt).to(device)
    f = pde(outt, tt, *param)

    loss_pde = mse(f, torch.zeros_like(f))
    loss_bc = mse(outt[0], torch.Tensor([x0, y0]))
    loss_total = lmbd * loss_bc + loss_pde

    if update_array:
        loss_pde_arr[epoch] = loss_pde
        loss_bc_arr[epoch] = loss_bc
        loss_arr[epoch] = loss_total

    return loss_total


def train(ep, parameters, lmbd):
    pbar = tqdm(range(ep), desc='Training Progress')
    for i in pbar:
        optimizer.zero_grad()
        loss = pdeloss(t, i, parameters[4], parameters[5], parameters[:4], lmbd)
        loss.backward()
        optimizer.step()


def load_parameters(file='lotka-volterra-parameters.txt'):
    with open(file) as f:
        res = []
        for i in f:
            res.append(float(i))
    return res


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prmtrs = load_parameters()
    t_start = int(prmtrs[0])
    t_end = int(prmtrs[1])
    t_steps = int(prmtrs[2])

    t = (torch.linspace(t_start, t_end, t_steps).unsqueeze(1)).to(device)
    t.requires_grad = True

    the_net = Net()

    train_switch = 1
    if train_switch:
        mse = nn.MSELoss()
        lr = 0.03
        optimizer = torch.optim.Adam(the_net.parameters(), lr=lr)

        epochs = 5000
        loss_arr = np.zeros(epochs)
        loss_pde_arr = np.zeros(epochs)
        loss_bc_arr = np.zeros(epochs)

        lambd = 10

        train(epochs, parameters=prmtrs[3:], lmbd=lambd)
        torch.save(the_net.state_dict(), 'Lotka–Volterra-weights.pth')

        print(f'first bc loss: {loss_bc_arr[0]}')
        print(f'first pde loss: {loss_pde_arr[0]}')
        print(f'average bc loss: {loss_bc_arr.mean()}')
        print(f'average pde loss: {loss_pde_arr.mean()}')
        print(f'last bc loss: {loss_bc_arr[-1]}')
        print(f'last pde loss: {loss_pde_arr[-1]}')

        plot_loss(loss_arr, loss_bc_arr, loss_pde_arr)

        out = the_net(t)
        plot_solution(t, out, 'Lotka–Volterra PINN approximation')
    else:
        the_net.load_state_dict(torch.load('Lotka–Volterra-weights.pth', weights_only=True))
        out = the_net(t)
        plot_solution(t, out, 'Lotka–Volterra PINN approximation')
