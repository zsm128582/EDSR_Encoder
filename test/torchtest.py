import torch

norm = torch.nn.LayerNorm(64,eps=1e-6)

x = torch.rand((16,224,64))
print(x.shape)
x = x[:,1:,:].mean(dim=1)
print(x.shape)
x = norm(x)
# x = x[:,0]
print(x.shape)
x = norm(x)
print(x.shape)
# cls_token = x[:,0]
# 64 - > 1000 
