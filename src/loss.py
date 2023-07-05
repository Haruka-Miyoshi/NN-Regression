import numpy as np
import matplotlib.pyplot as plt

loss=np.loadtxt('./model/loss.txt')
epoch=np.array([i for i in range(len(loss))])

plt.plot(epoch, loss, "b")
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./figs/loss.png')
plt.show()