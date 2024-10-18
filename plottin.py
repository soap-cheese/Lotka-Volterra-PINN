import matplotlib.pyplot as plt


def plot_loss_old(loss, lossbc, losspde, lmbd, save=False, title=''):
    plt.subplot(3, 1, 1)
    plt.title('Loss total')
    plt.plot(loss, label='loss total')
    # plt.plot(lossbc, label='loss bc')
    # plt.plot(losspde, label='loss pde')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(lossbc*lmbd, label='loss bc * lambda')
    plt.plot(losspde, label='loss pde')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(lossbc, label='loss bc')
    plt.plot(losspde, label='loss pde')
    plt.legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    if not save:
        plt.show()
    else:
        plt.savefig(title)


def plot_loss(log_loss, log_loss_bc, log_loss_pde, save=False, title=''):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.plot(log_loss, label='loss')
    plt.plot(log_loss_pde, label='loss pde')
    plt.plot(log_loss_bc, label='loss bc')
    plt.legend()

    if not save:
        plt.show()
    else:
        plt.savefig(title)


def plot_solution(tt, outt, title, save=False):

    plt.subplot(1, 2, 1)
    plt.title(title)

    plt.xlabel('t')
    try:
        plt.plot(tt, outt[:, 0], label='x (prey)')
    except RuntimeError:
        tt = tt.detach().numpy()
        outt = outt.detach().numpy()
        plt.plot(tt, outt[:, 0], label='x (prey)')

    plt.plot(tt, outt[:, 1], label='y (predator)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(outt[:, 0], outt[:, 1])

    if not save:
        plt.show()
    else:
        plt.savefig(title)

