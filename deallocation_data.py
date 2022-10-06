import torch 

a = torch.zeros(100,100,100).cuda()

print(torch.cuda.memory_allocated())

del a
torch.cuda.synchronize()
print(torch.cuda.memory_allocated())