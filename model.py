import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1=nn.Linear(100,128)
        self.dense2=nn.Linear(128,28*28)
        self.dropout1=nn.Dropout()

    def forward(self,z):
        fake=torch.relu(self.dense1(z))
        fake=self.dropout1(fake)
        fake=torch.tanh(self.dense2(fake))
        return fake

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1=nn.Linear(28*28,128)
        self.dense2=nn.Linear(128,1)
        self.dropout1=nn.Dropout()

    def forward(self,image):
        image=torch.relu(self.dense1(image))
        image=self.dropout1(image)
        prob=torch.sigmoid(self.dense2(image))
        return prob


