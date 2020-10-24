import torch
from model import Generator
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# モデル定義
generator = Generator()
model_path = 'trained_generator.pth'
generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

fake=generator(torch.rand((1,100)))
fake=fake.reshape(28,28)
fake=fake.detach().numpy()

plt.imshow((fake+1)/2,cmap="gray")
plt.show()
