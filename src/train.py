import os
import torch
import numpy as np
from nn import NeuralNetwork
import matplotlib.pyplot as plt

if not os.path.exists('./data'):
    os.mkdir('./data')

X=torch.load('./data/x.txt')
Y=torch.load('./data/y.txt')

x_rg=torch.load('./data/x_rang.txt')

nn_lr=NeuralNetwork()
nn_lr.update(X, Y, mode=True)

x=np.linspace(0,10,1000)
x1_new = np.sin(x)
x2_new = np.exp(x / 5)
x_new = np.stack([x1_new, x2_new], axis=1)

y=nn_lr.prediction(torch.tensor(x_new))

plt.plot(x, y.detach().numpy(), 'r')
plt.scatter(x_rg.numpy(), Y.numpy())
plt.xlabel('X')
plt.ylabel('Y')
plt.title('train')
plt.savefig('./figs/train.png')
plt.show()