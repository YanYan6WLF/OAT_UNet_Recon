import torch, time

x = torch.randn(5000, 5000, device='cuda')

# 
t0 = time.time()
y = torch.mm(x, x)
print("不加同步:", time.time() - t0)

# 
t0 = time.time()
y = torch.mm(x, x)
torch.cuda.synchronize()
print("加同步:", time.time() - t0)
