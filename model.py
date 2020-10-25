import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256,),
            nn.LeakyReLU(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        # 入力は(-1, 100)
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        prob = self.model(img)
        return prob

# import torchsummary
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,0,0.01)

# generator=Generator()
# generator.apply(init_weights)


# torchsummary.summary(generator, (28*28,))
