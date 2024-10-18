import torch
import os
from pinn import Net, load_parameters
from plottin import *
from rungekutta import runge_kutta


def do_stuff():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prmtrs = load_parameters()

    t = (torch.linspace(int(prmtrs[0]), int(prmtrs[1]), int(prmtrs[2])).unsqueeze(1)).to(device)
    t.requires_grad = True

    the_net = Net()

    the_net.load_state_dict(torch.load('Lotka–Volterra-weights.pth', weights_only=True))
    out_pinn = the_net(t)

    out_rk = runge_kutta(*prmtrs)

    t = t.detach().numpy()
    out_pinn = out_pinn.detach().numpy()
    plt.subplot(1, 2, 1)
    plt.plot(t, out_pinn, color='#f00')
    plt.plot(t, out_rk, color='#0f0')

    plt.subplot(1, 2, 2)
    plt.plot(out_pinn[:, 0], out_pinn[:, 1], color='#f00')
    plt.plot(out_rk[:, 0], out_rk[:, 1], color='#0f0')
    # os.mkdir('test')
    plt.savefig(f'test\\{int(os.listdir('test')[-1][0])+1}.jpg')

do_stuff()
