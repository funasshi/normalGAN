import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from model import Generator, Discriminator
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# データセットの定義 (訓練データの正規化、ラベルのone-hot-vec化)

transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: x.reshape((1, 28 * 28)),
                                lambda x: (x - 0.5) * 2])

target_transform = lambda x: torch.eye(10)[x]

trainset = torchvision.datasets.MNIST(root='datasets', train=True, download=True,
                                      transform=transform, target_transform=target_transform)
batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# ----------------------------------------------------------------------------------------------
# モデル定義
generator = Generator()
discriminator = Discriminator()

# 重みの初期化
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight,0,0.01)

# generator.apply(init_weights)
# discriminator.apply(init_weights)

#デバイス
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# discriminator.to(device)
# generator.to(device)

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    # discriminator = nn.DataParallel(discriminator)
    generator = generator.cuda()
    # generator = nn.DataParallel(generator)
    torch.backends.cudnn.benchmark = True

# ロス定義
loss_cross = nn.BCELoss()

# 最適化アルゴリズム定義
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
epochs = int(input("epochs:"))


def schedule_func(epoch):
    middle = epochs // 2
    if epoch < middle:
        return 1
    return (epochs - epoch) / middle


scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=schedule_func)
scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=schedule_func)


# ----------------------------------------------------------------------------------------------

# 訓練アルゴリズム
def train_discriminator(generator, discriminator, data):
    z = torch.rand((batch_size,1, 100))*2-1
    if torch.cuda.is_available():
        z = z.cuda()
        data = data.cuda()
    fake = generator(z)
    prob_fake = discriminator(fake)
    prob_data = discriminator(data)

    loss_A = loss_cross(prob_fake, torch.zeros_like(prob_fake))
    loss_B = loss_cross(prob_data, torch.ones_like(prob_data))
    loss_d = (loss_A + loss_B) / 2

    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    return loss_d.item()


def train_generator(generator, discriminator):
    z = torch.rand((batch_size,1, 100))*2-1
    if torch.cuda.is_available():
        z = z.cuda()
    fake = generator(z)
    prob_fake = discriminator(fake)

    loss_g = loss_cross(prob_fake, torch.ones_like(prob_fake))

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    return loss_g.item()


def train(generator, discriminator, epochs):
    loss_d_list = []
    loss_g_list = []
    for epoch in range(epochs):
        total_loss_d = 0
        total_loss_g = 0
        for i, (data, label) in enumerate(trainloader):
            total_loss_d += train_discriminator(generator, discriminator, data)
            total_loss_g += train_generator(generator, discriminator)
        total_loss_d /= (i + 1)
        total_loss_g /= (i + 1)
        loss_d_list.append(total_loss_d)
        loss_g_list.append(total_loss_g)
        print("epoch %3d : loss_g = %4f   loss_d = %4f" % (epoch, total_loss_g, total_loss_d))
        scheduler_g.step()
        scheduler_d.step()
    return loss_d_list, loss_g_list


# ----------------------------------------------------------------------------------------------
# 実際の処理
loss_d_list, loss_g_list = train(generator, discriminator, epochs)

# ----------------------------------------------------------------------------------------------
# 可視化
import numpy as np

x = np.arange(epochs)
plt.plot(x, loss_d_list, "r", label="d_loss")
plt.plot(x, loss_g_list, "g", label="g_loss")
plt.legend()
plt.savefig("loss.png")
# ----------------------------------------------------------------------------------------------
# generator保存
model_path = 'trained_generator.pth'
torch.save(generator.state_dict(), model_path)
