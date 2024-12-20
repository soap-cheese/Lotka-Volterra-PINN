import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from plottin import *
from rungekutta import runge_kutta
import sys
import os
import shutil


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Net(nn.Module):
    def __init__(self, hidden_size=16, activation=Sin):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(1, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),   # 1
            activation(),
            nn.Linear(hidden_size, hidden_size),   # 2
            activation(),
            nn.Linear(hidden_size, 2))

    def forward(self, x):
        return self.layers_stack(x)


def pde(outt, tt, alpha, beta, gamma, delta):
    x = outt[1:, 0].unsqueeze(1)
    y = outt[1:, 1].unsqueeze(1)
    dxdt = torch.autograd.grad(x, tt, torch.ones_like(x), create_graph=True,
                               retain_graph=True)[0][1:]
    dydt = torch.autograd.grad(y, tt, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0][1:]

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

    kutta_diff = mse(outt, kutta)

    if update_array:
        loss_pde_arr[epoch] = loss_pde
        loss_bc_arr[epoch] = loss_bc
        loss_arr[epoch] = loss_total
        kutta_diff_arr[epoch] = kutta_diff

    return loss_total


def train(ep, parameters, lmbd):
    pbar = tqdm(range(ep), desc='Training Progress')
    for i in pbar:
        optimizer.zero_grad()
        loss = pdeloss(t, i, parameters[4], parameters[5], parameters[:4], lmbd)
        loss.backward()
        optimizer.step()

        scheduler.step()
        pbar.set_postfix({'LR': scheduler.get_last_lr()[0]})


def load_parameters(file='lotka-volterra-parameters.txt'):
    '''
    t0
    tn
    n
    alpha
    beta
    gamma
    delta
    x0
    y0
    '''
    with open(file) as f:
        res = []
        for i in f:
            res.append(float(i))
    return res


def schedule(epoch):
    if epoch <= 200:
        return 10
    else:
        return 0.84**(epoch//500)

'''learning rate = 8 * 10**(-3) * schedule(epoch)'''


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prmtrs = load_parameters()
    t_start = prmtrs[0]
    t_end = prmtrs[1]
    t_steps = int(prmtrs[2])

    t = (torch.linspace(t_start, t_end, t_steps).unsqueeze(1)).to(device)
    t.requires_grad = True

    the_net = Net()

    train_switch = 1
    if train_switch:
        mse = nn.MSELoss()
        kutta = torch.from_numpy(runge_kutta(*prmtrs))
        default_lr, default_epochs, default_lambd = 8 * 10**(-3), 5000, 1
        while 1:
            i = input(': ')
            if i == 'exit': sys.exit()
            if not i:
                lr, epochs, lambd = default_lr, default_epochs, default_lambd
                the_net = Net()
            else:
                lr = input('lr: ')
                if not lr: lr = default_lr
                else: lr = float(lr)

                epochs = input('epochs: ')
                if not epochs: epochs = default_epochs
                else: epochs = int(epochs)

                lambd = input('lmbd: ')
                if not lambd: lambd = default_lambd
                else: lambd = float(lambd)

                if not bool(input('carry prev weights: ')): the_net = Net()

            loss_arr = np.zeros(epochs)
            loss_pde_arr = np.zeros(epochs)
            loss_bc_arr = np.zeros(epochs)
            kutta_diff_arr = np.zeros(epochs)

            optimizer = torch.optim.Adam(the_net.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

            # the_net.load_state_dict(torch.load('Lotka–Volterra-weights.pth', weights_only=True))
            train(epochs, parameters=prmtrs[3:], lmbd=lambd)

            cn = max([int(x.split(' ')[-1]) for x in os.listdir('cases')]) + 1
            os.mkdir(f"cases\\case {cn}")
            torch.save(the_net.state_dict(), f'cases\\case {cn}\\weights.pth')

            descr = f'lr = {lr}\nepochs = {epochs}\nlambda = {lambd}'
            # loss_bc = loss_bc_arr, loss_pde = loss_pde_arr,
            plot_loss(difference_with_numerical=kutta_diff_arr, loss=loss_arr,
                      save=f'cases\\case {cn}\\loss.jpg')

            out = the_net(t)
            plot_solution(t, out, '', save=f'cases\\case {cn}\\solution.jpg')
            with open(f'cases\\case {cn}\\net parameters.txt', 'w') as file:
                file.write(descr)
            shutil.copyfile('lotka-volterra-parameters.txt', f'cases\\case {cn}\\equation parameters.txt')

            print(f'last loss = {loss_arr[-1]}')
    else:
        the_net.load_state_dict(torch.load('Lotka–Volterra-weights.pth', weights_only=True))
        out = the_net(t)
        plot_solution(t, out, 'Lotka–Volterra PINN approximation')
