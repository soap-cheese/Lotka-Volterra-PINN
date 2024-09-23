import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def pde(outt, tt):
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


def pdeloss(tt, lmbd=1):
    outt = the_net(tt).to(device)
    f = pde(outt, tt)

    loss_pde = mse(f, torch.zeros_like(f))
    loss_bc = mse(outt[0], torch.Tensor([x0, y0]))

    return lmbd * loss_bc + loss_pde


def train(ep):
    pbar = tqdm(range(ep), desc='Training Progress')
    for i in pbar:
        optimizer.zero_grad()
        loss = pdeloss(t)
        loss_arr[i] = loss
        loss.backward()
        optimizer.step()


def plot_loss(loss):
    plt.title('Loss decreasing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(loss)
    plt.show()


def plot_solution(tt, outt):
    plt.title('Lotka–Volterra PINN approximation')
    plt.xlabel('t')
    plt.plot(tt.detach().numpy(), outt[:, 0].detach().numpy(), label='x (prey)')
    plt.plot(tt.detach().numpy(), outt[:, 1].detach().numpy(), label='y (predator)')
    plt.legend()
    plt.show()


def load_parameters(file='lotka-volterra-parameters.txt'):
    with open(file) as f:
        res = []
        for i in f:
            res.append(float(i))
    return res


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prmtrs = load_parameters()
t_start = int(prmtrs[0])
t_end = int(prmtrs[1])
t_steps = int(prmtrs[2])
alpha = prmtrs[3]
beta = prmtrs[4]
gamma = prmtrs[5]
delta = prmtrs[6]
x0 = prmtrs[7]
y0 = prmtrs[8]

t = (torch.linspace(t_start, t_end, t_steps).unsqueeze(1)).to(device)
t.requires_grad = True

the_net = Net()

train_switch = 0
if train_switch:
    mse = nn.MSELoss()
    lr = 0.01
    optimizer = torch.optim.Adam(the_net.parameters(), lr=lr)

    epochs = 2000
    loss_arr = np.zeros(epochs)

    train(epochs)
    torch.save(the_net.state_dict(), 'Lotka–Volterra-weights.pth')

    print(loss_arr[-1])
    plot_loss(loss_arr)
else:
    the_net.load_state_dict(torch.load('Lotka–Volterra-weights.pth', weights_only=True))
    out = the_net(t)
    plot_solution(t, out)
