import numpy as np
import matplotlib.pyplot as plt

def plot_lr_epoch(lrs):
    # for epoch, lr in enumerate(lrs):
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('LR Schedule')
    plt.grid(True)
    plt.show()

def save_lrs(param_groups,lrs):
    for i, param in enumerate(param_groups):
        lrs[i].append(param[i]['lr'])
    return lrs

def save_wbs(param_groups,wbs):
    wbs.append({
        'w_mu':param_groups[1]['params'].detach().numpy().cpu(),
        'b_mu':param_groups[2]['params'].detach().numpy().cpu(),
        'w_theta':param_groups[3]['params'].detach().numpy().cpu(),
        'b_theta':param_groups[4]['params'].detach().numpy().cpu(),
        'w_rho':param_groups[5]['params'].detach().numpy().cpu(),
        'b_rho':param_groups[6]['params'].detach().numpy().cpu(),
    })
    return wbs
