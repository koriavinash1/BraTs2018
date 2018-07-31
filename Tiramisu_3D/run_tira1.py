from tira1 import *
import torch

a= FCDenseNet57(2)

b= torch.autograd.Variable(torch.rand(2,3,64,64,64))

c= a(b)