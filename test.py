import matplotlib.pyplot as plt
from torch import nn,optim
import numpy as np

model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.9)

epochs=100
def s(epoch):
    middle=epochs//2
    if epoch<middle:
        return 1
    return (epochs-epoch)/middle

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = s)


optimizer_lr_list=[]
for epoch in range(0, 100): #ここは以下省略
    optimizer.step()
    scheduler.step()
    optimizer_lr_list.append(optimizer.param_groups[0]["lr"])

plt.plot(np.arange(100),optimizer_lr_list)
plt.show()