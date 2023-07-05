import os
import torch
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

if not os.path.exists('./data'):
    os.mkdir('./data')

# 一様分布からランダムに0-10の値を100回サンプリング
x_s=np.random.uniform(0, 10, 100)

x_sin=np.sin(x_s)
x_exp=np.exp(x_s/5)
x=np.stack([x_sin, x_exp], axis=1)
y=2*x_sin+2*x_exp+np.random.uniform(-1, 1, 100)

torch.save(torch.tensor(x_s), './data/x_rang.txt')
torch.save(torch.tensor(x), './data/x.txt')
torch.save(torch.tensor(y),'./data/y.txt')

plt.scatter(x_s, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('data')
plt.savefig('./figs/data.png')
plt.show()