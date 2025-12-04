from solver import solver
from Hyper_parameters import get_args
import os
from FISTA_NET import FISTANet
import torch 
from DataLoader import get_dataset
from metric import compute_measure

args=get_args()

# --------------------------------- device ---------------------------------------
device = torch.device("cude" if torch.cude.is_available() else "cpu")

# --------------------------- create an empty model ------------------------------
fista_net=FISTANet()
fista_net=fista_net.to(device)
fista_net_mode = 'train'
# -------------------------------- checkpoint -------------------------------------
if args.start_epoch > 0:
    model=os.path.join(args.save_model_path, 'epoch_{}.ckpt'.format(args.start_epoch))
    fista_net.load_state_dict(torch.load(model))

# ------------------------------------ solver / training ------------------------------
[train_loader, validate_loader]=get_dataset()

solver=solver(fista_net, trainloader=train_loader, validateloader=validate_loader)

# if fista_net_mode == 'train':
#     solver.train()
#     fista_net_test=solver.test()
# else:
#     fista_net_test=solver.test()



