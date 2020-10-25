import torch
from model import Generator
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# モデル定義
generator = Generator()
model_path = 'trained_generator.pth'
generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
generator.eval()


fig=plt.figure()
for i in range(1,21):
    ax=fig.add_subplot(4,5,i)
    fake=generator(torch.rand((1,100)))
    fake=fake.reshape(28,28)
    fake=fake.detach().numpy()
    ax.tick_params(labelbottom=False,
                   labelleft=False,
                   labelright=False,
                   labeltop=False)
    ax.tick_params(bottom=False,
                   left=False,
                   right=False,
                   top=False)
    ax.imshow((fake+1)/2,cmap="gray")

plt.savefig("generated_img.png")
plt.show()
