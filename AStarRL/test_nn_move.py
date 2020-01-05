import torch
from torch import nn
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt

b=1000
n=3

def get_data():
    X = torch.zeros((b,n))
    i = torch.randint(0, n-2, size=(b,))
    # i = random.randint(0, n-2)
    X[range(b), i] = torch.randint(0, n - 1, size=(b,), dtype=torch.float32)
    Y = X.clone().detach()
    Y[range(b), i], Y[range(b), i+1] = Y[range(b), i+1], Y[range(b), i]
    
    return X, Y

net = nn.Linear(n, n, bias=True)
net.weight.data = torch.zeros((n,n))
# net.bias.data = torch.zeros(n)
for i in range(n-1):
    net.weight.data[i+1][i] = 1
    

optim = torch.optim.Adam(net.parameters())
sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=.9)
loss = nn.MSELoss()

ll = []
plt.ion()
for epoch in range(10000):
    print(epoch)
    X, Y = get_data()
    y_pred = net(X)
    print(X, Y, y_pred)
    l = loss(y_pred, Y)
    print(l)
    
    optim.zero_grad()
    l.backward()
    optim.step()
    sched.step(epoch)
    
    ll.append(l.item())
    
    plt.plot(ll)
    # plt.draw()
    plt.pause(1e-6)
    plt.clf()