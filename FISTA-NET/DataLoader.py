# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       DataLoader.py
   Project Name:    Dataloader
   Author :         Yan
   Date:            2025/12/02
   Device:          Yan`s Laptop
-------------------------------------------------
   Change Activity:
                   2025/12/02:
-------------------------------------------------
"""

# import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import h5py 
from scipy.sparse import csc_matrix
from Hyper_parameters import get_args

'''
Args:
- train ratio=0.8
- self.size=size
- self.num_iter= num_iter
- self.tosize=tosize
'''
class DataSet(Dataset):

    def __init__(self, data_path,  train_ratio=0.8, mode='train',transforms_=None ): #  mode( train ration) data_path  transforms 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args=get_args()
        self.size=self.size
        self.tosize=self.tosize

        self.data_path = data_path
        self.files = self._get_file_list()
        num_total=len(self.files)
        num_train=int(num_total* train_ratio)

        if mode =='train':
            self.files=self.files[:num_train]
        elif mode =='validate':
            self.files=self.files[num_train:] 
        elif mode =='test':
            self.files=self.files

        print(f'{mode} dataset :{len(self.files)} samples')

        if transforms_:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None

        

    def _get_file_list(self):
        '''
        acquire all the data
        '''
        file_list=[]
        for root, dirs, files  in os.walk(self.data_path): # root: 我现在站在哪个文件夹里 dirs: 这个文件夹下有哪些子文件夹 files: 这个文件夹下有哪些文件
            for file in files:
                if file.endswith('_v73.mat'):
                    file_list.append(os.path.join(root,file))
        return sorted(file_list)




    def __getitem__(self, index):
        '''
        load:
        - A: forward matrix [len(b), img_size*img_size] ; CSC sparse matrix; different for each sample
        - b: detected single (M,)
        - FrangiFil: target/ ground truth (H,W); frangi filtered BP image
        - reconBP : backprojection reconstruction
        - reconMB_all: first 10 iteration of lSQR reconstruction
        '''
        with h5py.File(self.files[index % len(self.files)],'r') as mat_data:
            reconBP = np.array( mat_data['recon_het2'] )
            FrangiFil=np.array(mat_data['recon_BPhet2_Frangi_filter'])
            A_group = mat_data['A_mat'] 
            A_data=A_group['data'][:].flatten()# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            A_ir = A_group['ir'][:].flatten()
            A_jc = A_group['jc'][:].flatten()
            b=np.array(mat_data['sigMat_fil_MB']).reshape(-1) ## shape: (400, 512) flatten (204800,)

            A_csc=csc_matrix((A_data, A_ir,A_jc), shape=(204800, 15625)) # shape: (len(b), img_size*img_size) = (204800, 15625) numpy Compressed Sparse Column
            A_csr = A_csc.tocsr() # Compressed Sparse row 
            crow=A_csr.indptr
            col=A_csr.indices
            data=A_csr.data
            A_torch=torch.sparse_csr_tensor(
                torch.from_numpy(crow),
                torch.from_numpy(col),
                torch.from_numpy(data),
                size=A_csr.shape,
                device=self.device
            )

        # resize for maintain shape during training 
        # reconBP = cv2.resize(reconBP, (self.tosize, self.tosize))
        # FrangiFil = cv2.resize(FrangiFil, (self.tosize, self.tosize))


        # ---- Convert to tensor float32 ----
        reconBP=torch.from_numpy(reconBP).float()
        FrangiFil=torch.from_numpy(FrangiFil).float()
        # A_data=torch.from_numpy(A_data).float()
        # A_ir = torch.from_numpy(A_ir.astype(np.int64))
        # A_jc = torch.from_numpy(A_jc.astype(np.int64))
        b=torch.from_numpy(b).float()

        # -----Normalization-------
        reconBP = (reconBP - reconBP.min()) / (reconBP.max() - reconBP.min() + 1e-8)
        FrangiFil = (FrangiFil - FrangiFil.min()) / (FrangiFil.max() - FrangiFil.min() + 1e-8)

        # # ---- Channel Expansion ----
        reconBP=reconBP.unsqueeze(0) # [1, H, W]
        target= FrangiFil.unsqueeze(0) # [1, H, W]

        return {
            'b': b,
            'A':A_torch,
            # 'A_ir':A_ir,
            # 'A_jc':A_jc,
            'input': reconBP,
            'target':target
               }
    
    def __len__(self):
        return len(self.files)
    

def get_dataset():
    args=get_args()
    train_dataset = DataSet(data_path=args.data_path, train_ratio=0.8, mode='train',transforms_=None, size=args.size,tosize=args.tosize) #parameter
    validate_dataset = DataSet(data_path=args.data_path, train_ratio=0.8, mode='validate',transforms_=None, size=args.size,tosize=args.tosize)
    test_dataset = DataSet(data_path=args.test_data_path, train_ratio=0.8, mode='test',transforms_=None, size=args.size,tosize=args.tosize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= args.train_batch_size, shuffle=True,num_workers=0) # num_workers=0 safe for sparse matrix operation
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size= args.validate_batch_size, shuffle=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.test_batch_size, shuffle=False,num_workers=0)
    print(" Dataset Loading finished")
    return [train_loader, validate_loader,test_loader]
# for batch_idx, batch in enumerate((train_loader)): 